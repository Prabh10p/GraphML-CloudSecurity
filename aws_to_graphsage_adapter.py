#!/usr/bin/env python3
"""Read-only AWS inventory and conservative GraphSAGE graph adapter.

Requires: boto3, networkx. Optional inference requires torch, torch-geometric,
train_graphsage.py, build_graph_dataset.py and the trained checkpoint.

No AWS resource changes are made. Unknown facts are never silently set to safe.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
import networkx as nx
from botocore.exceptions import BotoCoreError, ClientError

NODE_TYPES = {"internet", "ec2", "iam", "s3"}
EDGE_TYPES = {"network_reachability", "uses_role", "permitted_access"}
REQUIRED = {"ec2": ("severity", "exposure"), "iam": ("privilege",), "s3": ("criticality",)}


def aws_call(fn, errors, context, default=None):
    try:
        return fn()
    except (ClientError, BotoCoreError) as exc:
        errors.append({"context": context, "error": str(exc)})
        return default


def discover(profile, region):
    session = boto3.Session(profile_name=profile, region_name=region)
    ec2 = session.client("ec2")
    iam = session.client("iam")
    s3 = session.client("s3")
    sts = session.client("sts")
    errors = []
    identity = aws_call(lambda: sts.get_caller_identity(), errors, "sts:GetCallerIdentity", {})
    account = identity.get("Account", "unknown")
    inventory = {"account": account, "region": region, "profile": profile,
                 "collected_at": datetime.now(timezone.utc).isoformat(),
                 "instances": [], "roles": [], "buckets": [], "errors": errors}

    # EC2 inventory: instance profile association is observable, not an attack path.
    paginator = ec2.get_paginator("describe_instances")
    pages = aws_call(lambda: list(paginator.paginate()), errors, "ec2:DescribeInstances", [])
    for page in pages:
        for reservation in page.get("Reservations", []):
            for instance in reservation.get("Instances", []):
                if instance.get("State", {}).get("Name") == "terminated":
                    continue
                inventory["instances"].append({
                    "id": instance["InstanceId"],
                    "public_ip_present": bool(instance.get("PublicIpAddress")),
                    "security_group_ids": [g["GroupId"] for g in instance.get("SecurityGroups", [])],
                    "instance_profile_arn": instance.get("IamInstanceProfile", {}).get("Arn"),
                    "state": instance.get("State", {}).get("Name"),
                })

    roles = aws_call(lambda: list(iam.get_paginator("list_roles").paginate()), errors, "iam:ListRoles", [])
    for page in roles:
        for role in page.get("Roles", []):
            inventory["roles"].append({"name": role["RoleName"], "arn": role["Arn"]})

    buckets = aws_call(lambda: s3.list_buckets(), errors, "s3:ListBuckets", {})
    for bucket in buckets.get("Buckets", []):
        inventory["buckets"].append({"name": bucket["Name"]})
    return inventory, session


def build_graph(inventory, annotations):
    """Annotations must be independently verified and explicitly supplied."""
    graph = nx.DiGraph()
    graph.graph.update({"source": "aws_read_only_with_manual_evidence", "schema_version": 1,
                        "account": inventory["account"], "region": inventory["region"],
                        "scenario_id": -1, "labeled": False})
    graph.add_node("internet", node_type="internet", severity=0.0, exposure=1,
                   privilege=0, criticality=0, risk_label=-1, evidence={"source": "synthetic_root"})
    known = {"internet"}
    for item in inventory["instances"]:
        nid = "ec2:" + item["id"]
        graph.add_node(nid, node_type="ec2", evidence={"aws_inventory": item})
        known.add(nid)
    for item in inventory["roles"]:
        nid = "iam:" + item["name"]
        graph.add_node(nid, node_type="iam", evidence={"aws_inventory": item})
        known.add(nid)
    for item in inventory["buckets"]:
        nid = "s3:" + item["name"]
        graph.add_node(nid, node_type="s3", evidence={"aws_inventory": item})
        known.add(nid)

    # Only explicit verified evidence creates edges. In particular, a public IP,
    # attached policy or trust policy alone does NOT establish exploitability.
    for nid, fields in annotations.get("nodes", {}).items():
        if nid not in known or nid == "internet":
            raise ValueError(f"Annotation references unknown node: {nid}")
        if not isinstance(fields, dict):
            raise ValueError(f"Node annotation must be an object: {nid}")
        allowed = set(REQUIRED[graph.nodes[nid]["node_type"]])
        for field, value in fields.items():
            if field not in allowed and field != "evidence":
                raise ValueError(f"Unsupported annotation field {field!r} on {nid}")
            if field == "evidence":
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"Nonempty evidence required: {nid}")
            elif field == "severity":
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 10:
                    raise ValueError(f"Severity must be 0..10: {nid}")
            elif type(value) is not int or value not in (0, 1):
                raise ValueError(f"{field} must be integer 0 or 1: {nid}")
            graph.nodes[nid][field] = value

    for edge in annotations.get("edges", []):
        src, dst = edge.get("source"), edge.get("target")
        relation, evidence = edge.get("relationship"), edge.get("evidence")
        if src not in known or dst not in known:
            raise ValueError(f"Edge has unknown endpoint: {src} -> {dst}")
        if relation not in EDGE_TYPES or not isinstance(evidence, str) or not evidence.strip():
            raise ValueError(f"Edge requires supported relationship and evidence: {edge}")
        expected = {"network_reachability": ("internet", "ec2"),
                    "uses_role": ("ec2", "iam"), "permitted_access": ("iam", "s3")}[relation]
        actual = (graph.nodes[src]["node_type"], graph.nodes[dst]["node_type"])
        if actual != expected:
            raise ValueError(f"Incorrect edge type: {src} -> {dst} ({relation})")
        graph.add_edge(src, dst, relationship=relation, evidence=evidence)

    unknown = []
    for nid, data in graph.nodes(data=True):
        if nid == "internet":
            continue
        for field in REQUIRED[data["node_type"]]:
            if field not in data:
                unknown.append({"node": nid, "field": field})
        if "evidence" not in annotations.get("nodes", {}).get(nid, {}):
            unknown.append({"node": nid, "field": "evidence"})
    return graph, unknown


def infer(graph, checkpoint_path):
    """Inference only for fully annotated graphs, never as validated AWS detection."""
    import torch
    from train_graphsage import GraphSAGE
    from build_graph_dataset import convert_graph

    model = GraphSAGE()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    threshold = float(checkpoint.get("decision_threshold", 0.5))

    # The existing converter expects these eight features and a risk_label.
    copy = graph.copy()
    for nid, data in copy.nodes(data=True):
        data["risk_label"] = -1 if nid == "internet" else 0
        if nid != "internet":
            data.setdefault("severity", 0.0)
            data.setdefault("exposure", 0)
            data.setdefault("privilege", 0)
            data.setdefault("criticality", 0)
    payload = nx.node_link_data(copy, edges="edges")
    tensor_graph = convert_graph(payload)
    with torch.no_grad():
        probs = torch.sigmoid(model(tensor_graph.x, tensor_graph.edge_index)).tolist()
    results = []
    for (nid, attrs), score in zip(copy.nodes(data=True), probs):
        if nid == "internet":
            continue
        results.append({"resource": nid, "resource_type": attrs["node_type"],
                        "model_score": round(float(score), 6),
                        "predicted_high_risk": bool(score >= threshold)})
    return {"threshold": threshold, "results": sorted(results, key=lambda x: -x["model_score"]),
            "warning": "Exploratory transfer inference only. Not validated vulnerability discovery or proof of access."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="cloud-project")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--annotations", type=Path, help="JSON of independently verified node/edge evidence")
    parser.add_argument("--output", type=Path, default=Path("results/aws_adapter"))
    parser.add_argument("--predict", action="store_true", help="Require complete annotations and run GraphSAGE")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/best_graphsage.pt"))
    args = parser.parse_args()

    annotations = {}
    if args.annotations:
        annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    if not isinstance(annotations, dict):
        parser.error("Annotations must be a JSON object")

    inventory, _ = discover(args.profile, args.region)
    graph, unknown = build_graph(inventory, annotations)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    (args.output / "graph.json").write_text(json.dumps(nx.node_link_data(graph, edges="edges"), indent=2), encoding="utf-8")
    (args.output / "unknown_fields.json").write_text(json.dumps(unknown, indent=2), encoding="utf-8")

    print(f"AWS inventory: {len(inventory['instances'])} EC2, {len(inventory['roles'])} IAM roles, "
          f"{len(inventory['buckets'])} S3 buckets")
    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} verified edges")
    print(f"Unknown or unevidenced fields: {len(unknown)}; AWS API errors: {len(inventory['errors'])}")
    print(f"Files saved to: {args.output}")
    if args.predict:
        if inventory["errors"]:
            parser.error("AWS inventory incomplete due to API errors; inference blocked")
        if unknown:
            parser.error("Incomplete annotations: inference blocked. See unknown_fields.json")
        if annotations.get("verified_complete") is not True:
            parser.error("Set verified_complete=true only after independently reviewing all required relationships; inference blocked")
        if graph.number_of_edges() == 0:
            parser.error("No verified relationships: inference blocked")
        if not args.checkpoint.is_file():
            parser.error(f"Checkpoint not found: {args.checkpoint}")
        report = infer(graph, args.checkpoint)
        (args.output / "predictions.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("Exploratory predictions saved to predictions.json")


if __name__ == "__main__":
    main()
