#!/usr/bin/env python3
"""Read-only AWS relationship evidence collector for GraphML-CloudSecurity.

Produces reviewable observations and candidate edges; never claims effective
permissions, exploitability, or full Internet reachability from partial evidence.
No AWS writes and no GraphSAGE inference.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

import boto3
from botocore.exceptions import BotoCoreError, ClientError


def attempt(fn, errors, operation, default=None):
    try:
        return fn()
    except (BotoCoreError, ClientError) as exc:
        errors.append({"operation": operation, "code": getattr(exc, "response", {}).get("Error", {}).get("Code", type(exc).__name__)})
        return default


def pages(client, operation, errors, **kwargs):
    try:
        return list(client.get_paginator(operation).paginate(**kwargs))
    except (BotoCoreError, ClientError) as exc:
        errors.append({"operation": operation, "code": getattr(exc, "response", {}).get("Error", {}).get("Code", type(exc).__name__)})
        return []


def collect(profile, region):
    session = boto3.Session(profile_name=profile, region_name=region)
    ec2, iam, s3 = (session.client(s) for s in ("ec2", "iam", "s3"))
    errors, instances, roles, buckets, security_groups, observations, candidates = [], [], [], [], [], [], []
    instance_pages = pages(ec2, "describe_instances", errors)
    group_pages = pages(ec2, "describe_security_groups", errors)
    groups = {g["GroupId"]: g for p in group_pages for g in p.get("SecurityGroups", [])}
    for group_id, g in groups.items():
        ingress = []
        for permission in g.get("IpPermissions", []):
            cidrs = [r["CidrIp"] for r in permission.get("IpRanges", [])]
            cidrs += [r["CidrIpv6"] for r in permission.get("Ipv6Ranges", [])]
            ingress.append({"protocol": permission.get("IpProtocol"), "from_port": permission.get("FromPort"),
                            "to_port": permission.get("ToPort"), "cidrs": cidrs,
                            "source_security_groups": [r.get("GroupId") for r in permission.get("UserIdGroupPairs", [])]})
        security_groups.append({"id": group_id, "vpc_id": g.get("VpcId"), "ingress": ingress})
    for page in instance_pages:
        for reservation in page.get("Reservations", []):
            for instance in reservation.get("Instances", []):
                if instance.get("State", {}).get("Name") == "terminated":
                    continue
                iid = instance["InstanceId"]
                profile_arn = instance.get("IamInstanceProfile", {}).get("Arn")
                sgids = [g["GroupId"] for g in instance.get("SecurityGroups", [])]
                public_ingress = []
                for gid in sgids:
                    for rule in next((g["ingress"] for g in security_groups if g["id"] == gid), []):
                        if any(c in ("0.0.0.0/0", "::/0") for c in rule["cidrs"]):
                            public_ingress.append({"security_group": gid, **rule})
                item = {"id": iid, "state": instance.get("State", {}).get("Name"),
                        "public_ip_present": bool(instance.get("PublicIpAddress")),
                        "vpc_id": instance.get("VpcId"), "subnet_id": instance.get("SubnetId"),
                        "security_group_ids": sgids, "instance_profile_arn": profile_arn,
                        "public_ingress_rules": public_ingress}
                instances.append(item)
                if public_ingress:
                    observations.append({"resource": f"ec2:{iid}", "finding": "security_group_has_world_cidr_ingress",
                                         "evidence": "ec2:DescribeSecurityGroups", "interpretation": "Not proof of Internet reachability or vulnerability"})
                if profile_arn:
                    profile_name = unquote(profile_arn.rsplit("/", 1)[-1])
                    response = attempt(lambda n=profile_name: iam.get_instance_profile(InstanceProfileName=n), errors,
                                       f"iam:GetInstanceProfile:{profile_name}", {})
                    for role in response.get("InstanceProfile", {}).get("Roles", []):
                        candidates.append({"source": f"ec2:{iid}", "target": f"iam:{role['RoleName']}",
                                           "relationship": "uses_role", "status": "observed_association",
                                           "evidence": f"EC2 instance profile {profile_arn}; IAM GetInstanceProfile role membership",
                                           "note": "Verified attachment, not proof of credential use or exploitability"})
    for p in pages(iam, "list_roles", errors):
        for role in p.get("Roles", []):
            name = role["RoleName"]
            attached = attempt(lambda n=name: iam.list_attached_role_policies(RoleName=n), errors,
                               f"iam:ListAttachedRolePolicies:{name}", {})
            inline = attempt(lambda n=name: iam.list_role_policies(RoleName=n), errors,
                             f"iam:ListRolePolicies:{name}", {})
            roles.append({"name": name, "arn": role["Arn"],
                          "attached_policy_arns": [p["PolicyArn"] for p in attached.get("AttachedPolicies", [])],
                          "inline_policy_names": inline.get("PolicyNames", []),
                          "policy_documents_not_evaluated": True})
    listing = attempt(lambda: s3.list_buckets(), errors, "s3:ListBuckets", {})
    for bucket in listing.get("Buckets", []):
        name = bucket["Name"]
        pab = attempt(lambda n=name: s3.get_public_access_block(Bucket=n), errors,
                      f"s3:GetPublicAccessBlock:{name}", {})
        policy_status = attempt(lambda n=name: s3.get_bucket_policy_status(Bucket=n), errors,
                                f"s3:GetBucketPolicyStatus:{name}", {})
        encryption = attempt(lambda n=name: s3.get_bucket_encryption(Bucket=n), errors,
                             f"s3:GetBucketEncryption:{name}", {})
        buckets.append({"name": name,
                        "public_access_block": pab.get("PublicAccessBlockConfiguration"),
                        "policy_public_status": policy_status.get("PolicyStatus", {}).get("IsPublic"),
                        "encryption_configuration_observed": bool(encryption),
                        "criticality": "unknown_requires_owner_label"})
    return {"schema_version": 1, "profile": profile, "region": region,
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "instances": instances, "security_groups": security_groups, "roles": roles, "buckets": buckets,
            "observations": observations, "candidate_edges": candidates, "api_errors": errors,
            "limitations": ["Only configured AWS region scanned for EC2 and security groups",
                            "IAM policy names and attachments do not establish effective S3 permissions",
                            "Public IP plus world-CIDR ingress does not prove Internet reachability",
                            "No vulnerability severity is inferred; use verified Inspector findings or documented lab evidence",
                            "S3 criticality must be independently labeled",
                            "No automatically generated permitted_access or network_reachability edges"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", default="cloud-project")
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--output", type=Path, default=Path("results/aws_adapter/relationship_evidence.json"))
    a = p.parse_args()
    report = collect(a.profile, a.region)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Read-only discovery complete")
    for field in ("instances", "security_groups", "roles", "buckets", "observations", "candidate_edges", "api_errors"):
        print(f"  {field}: {len(report[field])}")
    print("  Verified model-ready Internet/permission attack paths: not established")
    print(f"  Evidence saved: {a.output}")
    print("  Review API errors and unknowns before inference. Do not publish the raw evidence file.")


if __name__ == "__main__":
    main()
