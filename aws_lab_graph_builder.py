
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import boto3
import networkx as nx
from botocore.exceptions import ClientError


parser = argparse.ArgumentParser()
parser.add_argument("--profile", default="cloud-project")
parser.add_argument("--region", default="us-east-1")
args = parser.parse_args()

session = boto3.Session(
    profile_name=args.profile,
    region_name=args.region
)

ec2 = session.client("ec2")
iam = session.client("iam")
s3 = session.client("s3")

OUTPUT = Path("results/aws_lab")
OUTPUT.mkdir(parents=True, exist_ok=True)

GRAPH = nx.DiGraph()

INSTANCE_NAMES = [
    "graphml-secure-ec2",
    "graphml-risky-ec2"
]

BUCKETS = {
    "graphml-lab-critical-2026": 1,
    "graphml-lab-noncritical-2026": 0
}

ACTIONS = [
    "s3:GetObject",
    "s3:PutObject",
    "s3:DeleteObject"
]

permission_results = []
errors = []
roles = {}

# -------------------------------------------------
# 1. Discover research S3 buckets
# -------------------------------------------------

for bucket, criticality in BUCKETS.items():

    try:
        s3.head_bucket(Bucket=bucket)

        GRAPH.add_node(
            bucket,
            resource_type="s3",
            criticality=criticality,
            criticality_source="research_design"
        )

        print(f"[S3] {bucket}")

    except ClientError as e:
        errors.append({
            "resource": bucket,
            "operation": "HeadBucket",
            "error": e.response["Error"]["Code"]
        })


# -------------------------------------------------
# 2. Discover research EC2 instances
# -------------------------------------------------

paginator = ec2.get_paginator("describe_instances")

pages = paginator.paginate(
    Filters=[
        {
            "Name": "tag:Name",
            "Values": INSTANCE_NAMES
        },
        {
            "Name": "instance-state-name",
            "Values": ["running", "stopped"]
        }
    ]
)

for page in pages:
    for reservation in page["Reservations"]:
        for instance in reservation["Instances"]:

            tags = {
                tag["Key"]: tag["Value"]
                for tag in instance.get("Tags", [])
            }

            name = tags.get("Name")
            node_id = instance["InstanceId"]

            GRAPH.add_node(
                node_id,
                name=name,
                resource_type="ec2",
                public_ip_present=bool(
                    instance.get("PublicIpAddress")
                ),
                state=instance["State"]["Name"],
                vulnerability_severity=None,
                internet_reachability=None,
                s3_network_reachability=None
            )

            print(f"[EC2] {name}")

            # -------------------------------------
            # 3. Collect security group evidence
            # -------------------------------------

            group_ids = [
                group["GroupId"]
                for group in instance.get(
                    "SecurityGroups", []
                )
            ]

            inbound_rules = []
            outbound_rules = []

            try:
                if group_ids:
                    response = ec2.describe_security_groups(
                        GroupIds=group_ids
                    )

                    for group in response["SecurityGroups"]:

                        inbound_rules.extend(
                            group.get("IpPermissions", [])
                        )

                        outbound_rules.extend(
                            group.get("IpPermissionsEgress", [])
                        )

                GRAPH.nodes[node_id]["inbound_rules"] = (
                    inbound_rules
                )

                GRAPH.nodes[node_id]["outbound_rules"] = (
                    outbound_rules
                )

                GRAPH.nodes[node_id]["inbound_rule_count"] = (
                    len(inbound_rules)
                )

                GRAPH.nodes[node_id]["outbound_rule_count"] = (
                    len(outbound_rules)
                )

                print(
                    f"  Inbound rules: {len(inbound_rules)}"
                )

            except ClientError as e:
                errors.append({
                    "resource": name,
                    "operation": "DescribeSecurityGroups",
                    "error": e.response["Error"]["Code"]
                })

            # -------------------------------------
            # 4. Resolve instance profile to role
            # -------------------------------------

            profile = instance.get("IamInstanceProfile")

            if not profile:
                errors.append({
                    "resource": name,
                    "error": "MissingInstanceProfile"
                })
                continue

            profile_name = profile["Arn"].split("/")[-1]

            try:
                profile_data = iam.get_instance_profile(
                    InstanceProfileName=profile_name
                )["InstanceProfile"]

                for role in profile_data.get("Roles", []):

                    role_name = role["RoleName"]
                    role_arn = role["Arn"]

                    roles[role_name] = role_arn

                    GRAPH.add_node(
                        role_name,
                        resource_type="iam",
                        privilege=None
                    )

                    GRAPH.add_edge(
                        node_id,
                        role_name,
                        relationship="instance_profile_role",
                        evidence="GetInstanceProfile",
                        evidence_type="configuration"
                    )

                    print(f"  [IAM] {role_name}")

            except ClientError as e:
                errors.append({
                    "resource": name,
                    "operation": "GetInstanceProfile",
                    "error": e.response["Error"]["Code"]
                })


# -------------------------------------------------
# 5. Simulate IAM-to-S3 permissions
# -------------------------------------------------

for role_name, role_arn in roles.items():

    for bucket in BUCKETS:

        if bucket not in GRAPH:
            continue

        object_arn = f"arn:aws:s3:::{bucket}/test.txt"

        allowed_actions = []

        try:
            paginator = iam.get_paginator(
                "simulate_principal_policy"
            )

            evaluations = []

            for page in paginator.paginate(
                PolicySourceArn=role_arn,
                ActionNames=ACTIONS,
                ResourceArns=[object_arn]
            ):
                evaluations.extend(
                    page["EvaluationResults"]
                )

            for result in evaluations:

                action = result["EvalActionName"]
                decision = result["EvalDecision"]
                resource = result["EvalResourceName"]

                valid = (
                    resource == object_arn
                    and not result.get(
                        "MissingContextValues", []
                    )
                )

                permission_results.append({
                    "role": role_name,
                    "bucket": bucket,
                    "action": action,
                    "decision": decision,
                    "resource_validated": valid
                })

                if decision == "allowed" and valid:
                    allowed_actions.append(action)

            if allowed_actions:

                GRAPH.add_edge(
                    role_name,
                    bucket,
                    relationship="simulated_s3_permission",
                    actions=sorted(set(allowed_actions)),
                    tested_object="test.txt",
                    evidence="SimulatePrincipalPolicy",
                    evidence_type="policy_simulation"
                )

                print(
                    f"[EDGE] {role_name} -> {bucket}"
                )
                print(f"  Allowed: {allowed_actions}")

        except ClientError as e:
            errors.append({
                "resource": role_name,
                "bucket": bucket,
                "operation": "SimulatePrincipalPolicy",
                "error": e.response["Error"]["Code"]
            })


# -------------------------------------------------
# 6. Calculate permission-based risk features
# -------------------------------------------------

for role_name in roles:

    results = [
        result for result in permission_results
        if result["role"] == role_name
        and result["resource_validated"]
    ]

    allowed = [
        result for result in results
        if result["decision"] == "allowed"
    ]

    critical_access = any(
        BUCKETS[result["bucket"]] == 1
        for result in allowed
    )

    write_or_delete = any(
        result["action"] in [
            "s3:PutObject",
            "s3:DeleteObject"
        ]
        for result in allowed
    )

    GRAPH.nodes[role_name]["allowed_permission_count"] = (
        len(allowed)
    )

    expected = {
        (bucket, action)
        for bucket in BUCKETS
        for action in ACTIONS
    }

    observed = {
        (result["bucket"], result["action"])
        for result in results
    }

    complete = observed == expected

    GRAPH.nodes[role_name]["permission_checks_complete"] = (
        complete
    )

    GRAPH.nodes[role_name]["critical_asset_access"] = (
        critical_access if complete else None
    )

    GRAPH.nodes[role_name]["privilege"] = (
        int(critical_access or write_or_delete)
        if complete else None
    )


# -------------------------------------------------
# 7. Export graph and evidence
# -------------------------------------------------

graph_data = {
    "collected_at": datetime.now(timezone.utc).isoformat(),
    "region": args.region,
    "source": "real_aws_configuration_and_simulation",
    "nodes": [
        {"id": node, **attributes}
        for node, attributes in GRAPH.nodes(data=True)
    ],
    "edges": [
        {
            "source": source,
            "target": target,
            **attributes
        }
        for source, target, attributes
        in GRAPH.edges(data=True)
    ],
    "limitations": [
        "IAM simulation is not runtime S3 access.",
        "Internet reachability is not established.",
        "Software vulnerability severity is unknown.",
        "Criticality is assigned by research design.",
        "Graph is not yet compatible with the original model."
    ]
}

with open(OUTPUT / "aws_lab_graph.json", "w") as f:
    json.dump(graph_data, f, indent=2, default=str)

with open(OUTPUT / "permission_evidence.json", "w") as f:
    json.dump(permission_results, f, indent=2)

with open(OUTPUT / "api_errors.json", "w") as f:
    json.dump(errors, f, indent=2)

print("\n" + "=" * 50)
print("REAL AWS GRAPH DISCOVERY COMPLETE")
print("=" * 50)

print(f"Nodes: {GRAPH.number_of_nodes()}")
print(f"Edges: {GRAPH.number_of_edges()}")
print(f"Permission decisions: {len(permission_results)}")

print(
    "Allowed permissions:",
    sum(
        result["decision"] == "allowed"
        and result["resource_validated"]
        for result in permission_results
    )
)

print(f"API errors: {len(errors)}")
print(f"Results saved to: {OUTPUT}")
