
import networkx as nx
import random
import json
import time
from pathlib import Path

# ==========================================
# DATASET CONFIGURATION
# ==========================================

NUM_GRAPHS = 20000
SEED = 42
PROGRESS_INTERVAL = 1000

rng = random.Random(SEED)

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets"
OUTPUT_DIR = DATASET_DIR / "synthetic"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# GENERATE SYNTHETIC AWS CLOUD ENVIRONMENT
# ==========================================

def generate_cloud_graph(scenario_id):

    G = nx.DiGraph()

    # Internet entry point
    G.add_node(
        "internet",
        node_type="internet",
        severity=0.0,
        exposure=1,
        privilege=0,
        criticality=0
    )

    # Random environment size
    num_ec2 = rng.randint(5, 20)
    num_roles = rng.randint(5, 15)
    num_buckets = rng.randint(10, 30)

    # ======================================
    # GENERATE EC2 INSTANCES
    # ======================================

    for i in range(num_ec2):

        node = f"ec2_{i}"

        exposure = rng.randint(0, 1)
        severity = round(rng.uniform(0, 10), 2)
        criticality = rng.randint(0, 1)

        G.add_node(
            node,
            node_type="ec2",
            severity=severity,
            exposure=exposure,
            privilege=0,
            criticality=criticality
        )

        if exposure:
            G.add_edge(
                "internet",
                node,
                relationship="network_reachability"
            )

    # ======================================
    # GENERATE IAM ROLES
    # ======================================

    for i in range(num_roles):

        node = f"role_{i}"

        G.add_node(
            node,
            node_type="iam",
            severity=0.0,
            exposure=0,
            privilege=rng.randint(0, 1),
            criticality=0
        )

    # ======================================
    # GENERATE S3 BUCKETS
    # ======================================

    for i in range(num_buckets):

        node = f"s3_{i}"

        G.add_node(
            node,
            node_type="s3",
            severity=0.0,
            exposure=0,
            privilege=0,
            criticality=rng.randint(0, 1)
        )

    # ======================================
    # CONNECT EC2 INSTANCES TO IAM ROLES
    # ======================================

    for i in range(num_ec2):

        selected_role = rng.randrange(num_roles)

        G.add_edge(
            f"ec2_{i}",
            f"role_{selected_role}",
            relationship="uses_role"
        )

    # ======================================
    # CONNECT IAM ROLES TO S3 BUCKETS
    # ======================================

    for i in range(num_roles):

        role = f"role_{i}"
        privilege = G.nodes[role]["privilege"]

        # Broad permissions versus limited permissions
        if privilege:
            access_count = rng.randint(
                3, min(10, num_buckets)
            )
        else:
            access_count = 1

        selected_buckets = rng.sample(
            range(num_buckets),
            access_count
        )

        for bucket in selected_buckets:

            G.add_edge(
                role,
                f"s3_{bucket}",
                relationship="permitted_access"
            )

    # ======================================
    # SCENARIO METADATA
    # ======================================

    G.graph["scenario_id"] = scenario_id
    G.graph["source"] = "synthetic"
    G.graph["schema_version"] = 1
    G.graph["cloud_provider"] = "AWS"
    G.graph["labeled"] = False

    return G


# ==========================================
# GENERATE COMPLETE DATASET
# ==========================================

def main():

    rng.seed(SEED)
    start_time = time.time()

    total_nodes = 0
    total_edges = 0

    resource_counts = {
        "internet": 0,
        "ec2": 0,
        "iam": 0,
        "s3": 0
    }

    relationship_counts = {
        "network_reachability": 0,
        "uses_role": 0,
        "permitted_access": 0
    }

    exposed_ec2 = 0
    privileged_roles = 0
    critical_buckets = 0
    high_severity_ec2 = 0

    print("=" * 60)
    print("GRAPH ML CLOUD SECURITY")
    print("Synthetic AWS Dataset Generator")
    print("=" * 60)
    print(f"Target environments: {NUM_GRAPHS:,}")
    print(f"Random seed: {SEED}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)

    for scenario_id in range(NUM_GRAPHS):

        # Generate cloud environment
        G = generate_cloud_graph(scenario_id)

        # Convert NetworkX graph to JSON
        data = nx.node_link_data(G)

        # Save individual scenario
        file_path = OUTPUT_DIR / (
            f"scenario_{scenario_id:05d}.json"
        )

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, separators=(",", ":"))

        total_nodes += G.number_of_nodes()
        total_edges += G.number_of_edges()

        # Collect resource statistics
        for _, attributes in G.nodes(data=True):

            node_type = attributes["node_type"]

            resource_counts[node_type] += 1

            if node_type == "ec2":

                if attributes["exposure"] == 1:
                    exposed_ec2 += 1

                if attributes["severity"] >= 7.0:
                    high_severity_ec2 += 1

            if node_type == "iam":

                if attributes["privilege"] == 1:
                    privileged_roles += 1

            if node_type == "s3":

                if attributes["criticality"] == 1:
                    critical_buckets += 1

        # Collect relationship statistics
        for _, _, attributes in G.edges(data=True):

            relationship = attributes["relationship"]

            relationship_counts[relationship] += 1

        # Display progress
        completed = scenario_id + 1

        if (
            completed % PROGRESS_INTERVAL == 0
            or completed == NUM_GRAPHS
        ):

            elapsed = time.time() - start_time

            percentage = (
                completed / NUM_GRAPHS
            ) * 100

            print(
                f"Generated {completed:,}/{NUM_GRAPHS:,} "
                f"graphs | {percentage:.1f}% "
                f"| Elapsed: {elapsed:.1f}s"
            )

    # ======================================
    # DATASET SUMMARY
    # ======================================

    execution_time = time.time() - start_time

    summary = {
        "project": "GraphML-CloudSecurity",
        "cloud_provider": "AWS",
        "source": "synthetic",
        "schema_version": 1,
        "random_seed": SEED,
        "labeled": False,

        "total_graphs": NUM_GRAPHS,
        "total_nodes": total_nodes,
        "total_edges": total_edges,

        "resource_counts": resource_counts,

        "relationship_counts": relationship_counts,

        "security_statistics": {
            "internet_exposed_ec2": exposed_ec2,
            "high_severity_ec2": high_severity_ec2,
            "overprivileged_iam_roles": privileged_roles,
            "critical_s3_buckets": critical_buckets
        },

        "average_nodes_per_graph": round(
            total_nodes / NUM_GRAPHS, 2
        ),

        "average_edges_per_graph": round(
            total_edges / NUM_GRAPHS, 2
        ),

        "planned_dataset_split": {
            "training": 14000,
            "validation": 3000,
            "testing": 3000
        },

        "generation_time_seconds": round(
            execution_time, 2
        )
    }

    summary_path = DATASET_DIR / "dataset_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:

        json.dump(summary, file, indent=4)

    # ======================================
    # DISPLAY FINAL RESULTS
    # ======================================

    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETE")
    print("=" * 60)

    print(json.dumps(summary, indent=4))

    print("=" * 60)
    print(f"Total environments: {NUM_GRAPHS:,}")
    print(f"Total nodes: {total_nodes:,}")
    print(f"Total edges: {total_edges:,}")
    print(f"Generation time: {execution_time:.2f} seconds")
    print(f"Dataset location: {OUTPUT_DIR}")
    print(f"Summary location: {summary_path}")
    print("=" * 60)


# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":

    main()
