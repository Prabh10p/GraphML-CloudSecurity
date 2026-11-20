
import json
import time
import random
import networkx as nx

from pathlib import Path
from collections import Counter

# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets"

INPUT_DIR = DATASET_DIR / "synthetic"
OUTPUT_DIR = DATASET_DIR / "labeled"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
SEVERITY_THRESHOLD = 7.0
PROGRESS_INTERVAL = 1000


# ==========================================
# LOAD GRAPH
# ==========================================

def load_graph(file_path):

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    G = nx.DiGraph()
    G.graph.update(data.get("graph", {}))

    for node in data["nodes"]:

        attributes = dict(node)
        node_id = attributes.pop("id")

        G.add_node(node_id, **attributes)

    # Support different NetworkX versions
    edges = data.get("edges", data.get("links"))

    if edges is None:
        raise ValueError("Graph contains no edge list")

    for edge in edges:

        attributes = dict(edge)
        source = attributes.pop("source")
        target = attributes.pop("target")

        G.add_edge(source, target, **attributes)

    return G


# ==========================================
# SIMULATE HIGH-RISK ATTACK PATHS
# ==========================================

def find_attack_paths(G):

    attack_paths = set()

    if "internet" not in G:
        return []

    for ec2 in G.successors("internet"):

        attributes = G.nodes[ec2]

        if attributes["node_type"] != "ec2":
            continue

        if attributes["exposure"] != 1:
            continue

        if attributes["severity"] < SEVERITY_THRESHOLD:
            continue

        if G.edges["internet", ec2].get(
            "relationship"
        ) != "network_reachability":
            continue

        # Identify roles associated with EC2
        for role in G.successors(ec2):

            if G.nodes[role]["node_type"] != "iam":
                continue

            if G.edges[ec2, role].get(
                "relationship"
            ) != "uses_role":
                continue

            # Identify accessible S3 resources
            for bucket in G.successors(role):

                bucket_attributes = G.nodes[bucket]

                if bucket_attributes["node_type"] != "s3":
                    continue

                if G.edges[role, bucket].get(
                    "relationship"
                ) != "permitted_access":
                    continue

                if bucket_attributes["criticality"] != 1:
                    continue

                attack_paths.add(
                    ("internet", ec2, role, bucket)
                )

    return [list(path) for path in sorted(attack_paths)]


# ==========================================
# LABEL CLOUD RESOURCES
# ==========================================

def label_graph(G):

    attack_paths = find_attack_paths(G)

    path_counts = Counter()

    for path in attack_paths:

        for node in path:

            if node != "internet":
                path_counts[node] += 1

    exposed_critical_assets = {
        path[-1] for path in attack_paths
    }

    for node, attributes in G.nodes(data=True):

        if attributes["node_type"] == "internet":

            # Exclude Internet from supervised learning
            attributes["risk_label"] = -1
            attributes["attack_path_count"] = 0

            continue

        count = path_counts[node]

        attributes["attack_path_count"] = count

        attributes["risk_label"] = (
            1 if count > 0 else 0
        )

    G.graph["labeled"] = True
    G.graph["label_method"] = "simulated_attack_paths_v1"

    G.graph["attack_paths"] = attack_paths
    G.graph["total_attack_paths"] = len(attack_paths)

    G.graph["reachable_critical_assets"] = (
        len(exposed_critical_assets)
    )

    return G


# ==========================================
# SPLIT DATA BY COMPLETE SCENARIO
# ==========================================

def create_dataset_splits(scenario_files):

    rng = random.Random(SEED)

    names = sorted(scenario_files)
    rng.shuffle(names)

    total = len(names)

    train_end = int(total * 0.70)
    validation_end = train_end + int(total * 0.15)

    splits = {
        "train": names[:train_end],
        "validation": names[train_end:validation_end],
        "test": names[validation_end:]
    }

    for split_name, files in splits.items():

        split_directory = DATASET_DIR / split_name

        split_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        manifest = {
            "split": split_name,
            "total_graphs": len(files),
            "files": files
        }

        with open(
            split_directory / "manifest.json",
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(manifest, file, indent=2)

    return {
        name: len(files)
        for name, files in splits.items()
    }


# ==========================================
# LABEL COMPLETE DATASET
# ==========================================

def main():

    start_time = time.time()

    files = sorted(
        INPUT_DIR.glob("scenario_*.json")
    )

    if len(files) != 20000:

        raise RuntimeError(
            f"Expected 20,000 scenarios, found {len(files)}. "
            "Check the synthetic dataset directory."
        )

    total_graphs = len(files)

    total_nodes = 0
    total_edges = 0
    high_risk_nodes = 0
    negative_nodes = 0
    excluded_nodes = 0

    total_attack_paths = 0
    graphs_with_attack_paths = 0
    total_exposed_critical_assets = 0

    risk_by_type = Counter()
    resource_counts = Counter()

    processed_files = []
    seen_scenario_ids = set()

    print("=" * 60)
    print("GRAPH ML CLOUD SECURITY")
    print("Ground-Truth Labeling")
    print("=" * 60)
    print(f"Total scenarios: {total_graphs:,}")
    print(f"Severity threshold: {SEVERITY_THRESHOLD}")
    print("=" * 60)

    for index, file_path in enumerate(files):

        G = load_graph(file_path)

        scenario_id = G.graph["scenario_id"]

        if scenario_id in seen_scenario_ids:

            raise ValueError(
                f"Duplicate scenario ID: {scenario_id}"
            )

        seen_scenario_ids.add(scenario_id)

        G = label_graph(G)

        total_nodes += G.number_of_nodes()
        total_edges += G.number_of_edges()

        attack_count = G.graph["total_attack_paths"]

        total_attack_paths += attack_count

        total_exposed_critical_assets += (
            G.graph["reachable_critical_assets"]
        )

        if attack_count > 0:
            graphs_with_attack_paths += 1

        for node, attributes in G.nodes(data=True):

            label = attributes["risk_label"]
            node_type = attributes["node_type"]

            resource_counts[node_type] += 1

            if label == 1:

                high_risk_nodes += 1
                risk_by_type[node_type] += 1

            elif label == 0:

                negative_nodes += 1

            else:

                excluded_nodes += 1

        output_file = OUTPUT_DIR / file_path.name

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                nx.node_link_data(G),
                file,
                separators=(",", ":")
            )

        processed_files.append(file_path.name)

        completed = index + 1

        if (
            completed % PROGRESS_INTERVAL == 0
            or completed == total_graphs
        ):

            percentage = (
                completed / total_graphs
            ) * 100

            print(
                f"Labeled {completed:,}/{total_graphs:,} "
                f"| {percentage:.1f}%"
            )

    # Create independent scenario-level splits
    splits = create_dataset_splits(processed_files)

    supervised_nodes = (
        high_risk_nodes + negative_nodes
    )

    summary = {
        "project": "GraphML-CloudSecurity",
        "source": "synthetic",
        "labeled": True,

        "label_method": "simulated_attack_paths_v1",

        "severity_threshold": SEVERITY_THRESHOLD,

        "total_graphs": total_graphs,
        "total_nodes": total_nodes,
        "total_edges": total_edges,

        "high_risk_nodes": high_risk_nodes,
        "negative_nodes": negative_nodes,
        "excluded_internet_nodes": excluded_nodes,

        "high_risk_percentage": round(
            100 * high_risk_nodes
            / max(1, supervised_nodes),
            2
        ),

        "total_attack_paths": total_attack_paths,

        "graphs_with_attack_paths": (
            graphs_with_attack_paths
        ),

        "graphs_without_attack_paths": (
            total_graphs - graphs_with_attack_paths
        ),

        "total_reachable_critical_assets": (
            total_exposed_critical_assets
        ),

        "resource_counts": dict(resource_counts),
        "high_risk_by_resource_type": dict(risk_by_type),

        "dataset_splits": splits,

        "execution_time_seconds": round(
            time.time() - start_time,
            2
        )
    }

    summary_path = (
        DATASET_DIR / "labeling_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(summary, file, indent=4)

    print("\n" + "=" * 60)
    print("DATASET LABELING COMPLETE")
    print("=" * 60)

    print(json.dumps(summary, indent=4))

    print("=" * 60)
    print(f"Labeled dataset: {OUTPUT_DIR}")
    print(f"Summary: {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
