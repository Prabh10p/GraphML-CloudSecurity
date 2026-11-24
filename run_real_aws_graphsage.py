
import json
from pathlib import Path

import torch
import pandas as pd
import matplotlib.pyplot as plt

from train_graphsage import GraphSAGE


# ==========================================
# CONFIGURATION
# ==========================================

GRAPH_PATH = Path(
    "results/aws_lab/aws_lab_graph.json"
)

MODEL_PATH = Path(
    "models/best_graphsage.pt"
)

OUTPUT = Path(
    "results/aws_lab/inference"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model = GraphSAGE().to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

threshold = checkpoint["decision_threshold"]

print("=" * 65)
print("REAL AWS GRAPHSAGE INFERENCE")
print("=" * 65)

print("Model epoch:", checkpoint["epoch"])
print("Threshold:", threshold)


# ==========================================
# LOAD REAL AWS GRAPH
# ==========================================

with open(GRAPH_PATH) as f:
    graph = json.load(f)

nodes = graph["nodes"]
edges = graph["edges"]

node_index = {
    node["id"]: i
    for i, node in enumerate(nodes)
}

print("Real AWS nodes:", len(nodes))
print("Real AWS edges:", len(edges))


# ==========================================
# BUILD GRAPH CONNECTIVITY
# ==========================================

edge_pairs = []

for edge in edges:

    source = node_index[edge["source"]]
    target = node_index[edge["target"]]

    # Match bidirectional message passing
    # used during synthetic model training.

    edge_pairs.append([source, target])
    edge_pairs.append([target, source])

edge_index = torch.tensor(
    edge_pairs,
    dtype=torch.long
).t().contiguous()

if not edge_pairs:
    edge_index = torch.empty(
        (2, 0),
        dtype=torch.long
    )


# ==========================================
# FEATURE ENGINEERING
# ==========================================

FEATURE_NAMES = [
    "is_internet",
    "is_ec2",
    "is_iam",
    "is_s3",
    "severity_normalized",
    "exposure",
    "privilege",
    "criticality"
]


def build_features(assumed_severity):

    features = []

    for node in nodes:

        resource_type = node["resource_type"]

        is_internet = int(
            resource_type == "internet"
        )

        is_ec2 = int(resource_type == "ec2")
        is_iam = int(resource_type == "iam")
        is_s3 = int(resource_type == "s3")

        severity = 0.0
        exposure = 0.0
        privilege = 0.0
        criticality = 0.0

        if is_ec2:

            observed_severity = node.get(
                "vulnerability_severity"
            )

            if observed_severity is None:
                severity = assumed_severity / 10.0
            else:
                severity = float(
                    observed_severity
                ) / 10.0

            # Exploratory assumption for private lab.
            # Not a verified reachability assessment.
            exposure = 0.0

        if is_iam:

            value = node.get("privilege")

            if value is None:
                raise ValueError(
                    "Unknown IAM privilege feature"
                )

            privilege = float(value)

        if is_s3:

            value = node.get("criticality")

            if value is None:
                raise ValueError(
                    "Unknown asset criticality"
                )

            criticality = float(value)

        features.append([
            is_internet,
            is_ec2,
            is_iam,
            is_s3,
            severity,
            exposure,
            privilege,
            criticality
        ])

    return torch.tensor(
        features,
        dtype=torch.float32
    )


# ==========================================
# SENSITIVITY EXPERIMENTS
# ==========================================

severity_scenarios = [0, 3, 5, 7, 9, 10]

results = []

for assumed_severity in severity_scenarios:

    x = build_features(assumed_severity)

    with torch.no_grad():

        logits = model(
            x.to(DEVICE),
            edge_index.to(DEVICE)
        )

        scores = torch.sigmoid(
            logits
        ).cpu().numpy()

    print("\n" + "-" * 65)

    print(
        "Hypothetical EC2 severity:",
        assumed_severity
    )

    print("-" * 65)

    for node, score in zip(nodes, scores):

        name = node.get("name", node["id"])

        prediction = int(score >= threshold)

        results.append({
            "resource": name,
            "resource_type": node["resource_type"],
            "assumed_ec2_severity": assumed_severity,
            "assumed_internet_exposure": 0,
            "model_score": float(score),
            "threshold": threshold,
            "prediction": prediction,
            "evaluation_type": "exploratory_ood"
        })

        print(
            f"{name:40s} "
            f"Score: {score:.6f} "
            f"Prediction: {prediction}"
        )


# ==========================================
# SAVE RESULTS
# ==========================================

df = pd.DataFrame(results)

df.to_csv(
    OUTPUT / "real_aws_predictions.csv",
    index=False
)

summary = {
    "model_epoch": checkpoint["epoch"],
    "threshold": threshold,
    "real_aws_nodes": len(nodes),
    "real_aws_edges": len(edges),
    "severity_scenarios": severity_scenarios,
    "total_predictions": len(results),
    "evaluation_type": "exploratory_ood",
    "limitations": [
        "Original model trained on synthetic data.",
        "Unknown EC2 severity tested hypothetically.",
        "Internet exposure assumed zero.",
        "No Internet source node in real lab graph.",
        "Internal exposure not represented.",
        "IAM permissions based on policy simulation.",
        "Scores are not calibrated real-world probabilities."
    ],
    "predictions": results
}

with open(
    OUTPUT / "real_aws_inference.json",
    "w"
) as f:
    json.dump(summary, f, indent=2)


# ==========================================
# VISUALIZE MODEL SCORES
# ==========================================

plt.figure(figsize=(12, 7))

for resource, group in df.groupby("resource"):

    plt.plot(
        group["assumed_ec2_severity"],
        group["model_score"],
        marker="o",
        linewidth=2,
        label=resource
    )

plt.axhline(
    threshold,
    linestyle="--",
    label=f"Frozen threshold ({threshold})"
)

plt.xlabel("Hypothetical EC2 vulnerability severity")
plt.ylabel("GraphSAGE model score")

plt.title(
    "Exploratory GraphSAGE Inference on Real AWS\n"
    "Sensitivity to Assumed Vulnerability Severity"
)

plt.ylim(-0.02, 1.02)
plt.grid(alpha=0.25)
plt.legend(fontsize=8)
plt.tight_layout()

plt.savefig(
    OUTPUT / "real_aws_sensitivity.png",
    dpi=300
)

plt.close()

print("\n" + "=" * 65)
print("REAL AWS INFERENCE COMPLETE")
print("=" * 65)

print("Total predictions:", len(results))
print("Results saved to:", OUTPUT)