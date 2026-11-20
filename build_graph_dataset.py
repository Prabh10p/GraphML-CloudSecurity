
import json
import time
import torch

from pathlib import Path
from collections import Counter
from torch_geometric.data import Data

# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "datasets"
INPUT_DIR = DATASET_DIR / "labeled"
OUTPUT_DIR = DATASET_DIR / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NODE_TYPES = {
    "internet": 0,
    "ec2": 1,
    "iam": 2,
    "s3": 3
}

RELATIONSHIP_TYPES = {
    "network_reachability": 0,
    "uses_role": 1,
    "permitted_access": 2
}

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

PROGRESS_INTERVAL = 1000


# ==========================================
# LOAD JSON GRAPH
# ==========================================

def load_json(file_path):

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ==========================================
# CONVERT GRAPH TO PYTORCH GEOMETRIC
# ==========================================

def convert_graph(graph_data):

    nodes = graph_data["nodes"]

    edges = graph_data.get(
        "edges",
        graph_data.get("links")
    )

    if edges is None:
        raise ValueError("Missing graph edges")

    node_mapping = {
        node["id"]: index
        for index, node in enumerate(nodes)
    }

    features = []
    labels = []
    node_types = []

    # --------------------------------------
    # CREATE NODE FEATURES
    # --------------------------------------

    for node in nodes:

        node_type = NODE_TYPES[node["node_type"]]

        # One-hot encode resource type
        type_features = [0.0] * len(NODE_TYPES)
        type_features[node_type] = 1.0

        node_features = type_features + [
            float(node["severity"]) / 10.0,
            float(node["exposure"]),
            float(node["privilege"]),
            float(node["criticality"])
        ]

        features.append(node_features)

        labels.append(int(node["risk_label"]))

        node_types.append(node_type)

    x = torch.tensor(
        features,
        dtype=torch.float32
    )

    y = torch.tensor(
        labels,
        dtype=torch.long
    )

    # --------------------------------------
    # CREATE GRAPH RELATIONSHIPS
    # --------------------------------------

    forward_edges = []
    forward_types = []

    for edge in edges:

        source = node_mapping[edge["source"]]
        target = node_mapping[edge["target"]]

        relationship = RELATIONSHIP_TYPES[
            edge["relationship"]
        ]

        forward_edges.append([source, target])
        forward_types.append(relationship)

    # Preserve original directed relationships
    directed_edge_index = torch.tensor(
        forward_edges,
        dtype=torch.long
    ).reshape(-1, 2).t().contiguous()

    # Add reverse edges for GNN message passing.
    # Reverse edges are NOT real AWS permissions.

    message_edges = []
    message_types = []

    for (source, target), relationship in zip(
        forward_edges,
        forward_types
    ):

        message_edges.append([source, target])
        message_types.append(relationship)

        message_edges.append([target, source])
        message_types.append(
            relationship + len(RELATIONSHIP_TYPES)
        )

    edge_index = torch.tensor(
        message_edges,
        dtype=torch.long
    ).reshape(-1, 2).t().contiguous()

    edge_type = torch.tensor(
        message_types,
        dtype=torch.long
    )

    # --------------------------------------
    # CREATE SUPERVISION MASK
    # --------------------------------------

    # Internet nodes have label -1.
    # They participate in message passing,
    # but are excluded from model loss.

    supervised_mask = y >= 0

    scenario_id = graph_data["graph"]["scenario_id"]

    data = Data(
        x=x,
        edge_index=edge_index,
        directed_edge_index=directed_edge_index,
        edge_type=edge_type,
        y=y,
        supervised_mask=supervised_mask,
        node_type=torch.tensor(
            node_types,
            dtype=torch.long
        ),
        scenario_id=torch.tensor(
            [scenario_id],
            dtype=torch.long
        )
    )

    data.validate(raise_on_error=True)

    return data


# ==========================================
# PROCESS EACH DATASET SPLIT
# ==========================================

def process_split(split_name):

    manifest_path = (
        DATASET_DIR
        / split_name
        / "manifest.json"
    )

    manifest = load_json(manifest_path)

    files = manifest["files"]

    graph_objects = []

    total_nodes = 0
    original_edges = 0
    message_edges = 0

    label_counts = Counter()

    scenario_ids = set()

    print("\n" + "=" * 60)
    print(f"Processing {split_name.upper()} dataset")
    print(f"Graphs: {len(files):,}")
    print("=" * 60)

    for index, filename in enumerate(files):

        graph_data = load_json(
            INPUT_DIR / filename
        )

        data = convert_graph(graph_data)

        scenario_id = int(
            data.scenario_id.item()
        )

        if scenario_id in scenario_ids:

            raise ValueError(
                f"Duplicate scenario: {scenario_id}"
            )

        scenario_ids.add(scenario_id)

        graph_objects.append(data)

        total_nodes += data.num_nodes

        original_edges += (
            data.directed_edge_index.size(1)
        )

        message_edges += data.num_edges

        label_counts.update(data.y.tolist())

        completed = index + 1

        if (
            completed % PROGRESS_INTERVAL == 0
            or completed == len(files)
        ):

            print(
                f"Processed {completed:,}"
                f"/{len(files):,} graphs"
            )

    output_file = (
        OUTPUT_DIR / f"{split_name}.pt"
    )

    torch.save(
        graph_objects,
        output_file
    )

    supervised_nodes = (
        label_counts[0] + label_counts[1]
    )

    summary = {
        "split": split_name,
        "graphs": len(graph_objects),
        "nodes": total_nodes,
        "original_directed_edges": original_edges,
        "message_passing_edges": message_edges,
        "positive_nodes": label_counts[1],
        "negative_nodes": label_counts[0],
        "excluded_nodes": label_counts[-1],
        "supervised_nodes": supervised_nodes,
        "positive_percentage": round(
            100 * label_counts[1]
            / max(1, supervised_nodes),
            2
        ),
        "file_size_mb": round(
            output_file.stat().st_size
            / (1024 ** 2),
            2
        )
    }

    print(json.dumps(summary, indent=4))

    return summary, scenario_ids


# ==========================================
# MAIN
# ==========================================

def main():

    start_time = time.time()

    print("=" * 60)
    print("GRAPH ML CLOUD SECURITY")
    print("PyTorch Geometric Dataset Builder")
    print("=" * 60)

    print(f"PyTorch version: {torch.__version__}")
    print(f"Node features: {len(FEATURE_NAMES)}")

    summaries = {}

    all_scenario_ids = set()

    for split_name in [
        "train",
        "validation",
        "test"
    ]:

        summary, scenario_ids = process_split(
            split_name
        )

        if all_scenario_ids.intersection(
            scenario_ids
        ):

            raise RuntimeError(
                "Scenario overlap detected "
                "between dataset splits"
            )

        all_scenario_ids.update(scenario_ids)

        summaries[split_name] = summary

    # Calculate class weight from training data only

    train_positive = summaries[
        "train"
    ]["positive_nodes"]

    train_negative = summaries[
        "train"
    ]["negative_nodes"]

    if train_positive == 0 or train_negative == 0:
        raise RuntimeError(
            "Training split must contain both classes"
        )

    positive_class_weight = (
        train_negative / train_positive
    )

    final_summary = {
        "project": "GraphML-CloudSecurity",
        "format": "PyTorch Geometric",
        "total_graphs": len(all_scenario_ids),
        "number_of_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "node_types": NODE_TYPES,
        "relationship_types": RELATIONSHIP_TYPES,
        "number_of_message_edge_types": 6,
        "number_of_classes": 2,
        "scenario_overlap": False,
        "training_positive_class_weight": round(
            positive_class_weight,
            4
        ),
        "splits": summaries,
        "execution_time_seconds": round(
            time.time() - start_time,
            2
        )
    }

    summary_path = (
        OUTPUT_DIR / "processed_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            final_summary,
            f,
            indent=4
        )

    print("\n" + "=" * 60)
    print("GRAPH DATASET PREPARATION COMPLETE")
    print("=" * 60)

    print(json.dumps(final_summary, indent=4))

    print("=" * 60)
    print(f"Saved datasets: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
