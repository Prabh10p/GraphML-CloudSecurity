
import json
import copy
import time
import networkx as nx
import torch
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from train_graphsage import (
    GraphSAGE,
    select_device
)

from build_graph_dataset import convert_graph

from label_scenarios import (
    load_graph,
    find_attack_paths
)

# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "datasets"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

ACTION_COSTS = {
    "restrict_exposure": 2.0,
    "patch_vulnerability": 5.0,
    "revoke_s3_permission": 3.0
}


# ==========================================
# LOAD TRAINED AI MODEL
# ==========================================

def load_model():

    device = select_device()

    checkpoint = torch.load(
        MODEL_DIR / "best_graphsage.pt",
        map_location=device,
        weights_only=True
    )

    model = GraphSAGE().to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return (
        model,
        device,
        checkpoint["decision_threshold"]
    )


# ==========================================
# AI GRAPH RISK PREDICTION
# ==========================================

@torch.no_grad()
def predict_graph_risk(
    graph,
    model,
    device,
    threshold
):

    # The converter requires a label field.
    # Supply placeholder values, NOT answers.

    graph_data = nx.node_link_data(
        graph,
        edges="edges"
    )

    for node in graph_data["nodes"]:

        node["risk_label"] = 0

        if node["node_type"] == "internet":

            node["risk_label"] = -1

    data = convert_graph(graph_data)

    data = data.to(device)

    logits = model(
        data.x,
        data.edge_index
    )

    scores = torch.sigmoid(
        logits
    ).cpu().tolist()

    predictions = {}

    for index, node in enumerate(
        graph_data["nodes"]
    ):

        if node["node_type"] == "internet":
            continue

        predictions[node["id"]] = {

            "resource_type":
                node["node_type"],

            "risk_score":
                float(scores[index]),

            "predicted_high_risk":
                scores[index] >= threshold
        }

    return predictions


# ==========================================
# MEASURE GRAPH SECURITY RISK
# ==========================================

def measure_risk(
    graph,
    model,
    device,
    threshold
):

    predictions = predict_graph_risk(
        graph,
        model,
        device,
        threshold
    )

    # Independent rule-based simulator.
    # These results are NOT model features.

    attack_paths = find_attack_paths(
        graph
    )

    reachable_assets = {
        path[-1]
        for path in attack_paths
    }

    high_risk_nodes = {
        node
        for path in attack_paths
        for node in path[1:]
    }

    return {

        "ai_risk_score": sum(
            item["risk_score"]
            for item in predictions.values()
        ),

        "ai_high_risk_nodes": sum(
            item["predicted_high_risk"]
            for item in predictions.values()
        ),

        "attack_paths": len(
            attack_paths
        ),

        "reachable_critical_assets": len(
            reachable_assets
        ),

        "simulated_high_risk_nodes": len(
            high_risk_nodes
        )
    }


# ==========================================
# GENERATE REMEDIATION CANDIDATES
# ==========================================

def generate_actions(graph):

    actions = []

    for node, attributes in graph.nodes(
        data=True
    ):

        if attributes["node_type"] != "ec2":
            continue

        if attributes["exposure"] == 1:

            actions.append({

                "action_type":
                    "restrict_exposure",

                "target": node,

                "cost": ACTION_COSTS[
                    "restrict_exposure"
                ]
            })

        if attributes["severity"] >= 7.0:

            actions.append({

                "action_type":
                    "patch_vulnerability",

                "target": node,

                "cost": ACTION_COSTS[
                    "patch_vulnerability"
                ]
            })

    for source, target, attributes in (
        graph.edges(data=True)
    ):

        if (
            attributes.get("relationship")
            == "permitted_access"
        ):

            actions.append({

                "action_type":
                    "revoke_s3_permission",

                "source": source,

                "target": target,

                "cost": ACTION_COSTS[
                    "revoke_s3_permission"
                ]
            })

    return actions


# ==========================================
# SIMULATE A COUNTERFACTUAL INTERVENTION
# ==========================================

def apply_intervention(graph, action):

    modified = copy.deepcopy(graph)

    action_type = action["action_type"]
    target = action["target"]

    if action_type == "restrict_exposure":

        modified.nodes[target]["exposure"] = 0

        incoming_edges = list(
            modified.in_edges(
                target,
                data=True
            )
        )

        for source, destination, attributes in (
            incoming_edges
        ):

            if (
                modified.nodes[source]["node_type"]
                == "internet"
                and attributes.get("relationship")
                == "network_reachability"
            ):

                modified.remove_edge(
                    source,
                    destination
                )

    elif action_type == "patch_vulnerability":

        modified.nodes[target]["severity"] = 0.0

    elif action_type == "revoke_s3_permission":

        source = action["source"]

        if modified.has_edge(source, target):

            modified.remove_edge(
                source,
                target
            )

    else:

        raise ValueError(
            f"Unknown action: {action_type}"
        )

    return modified


# ==========================================
# EVALUATE EACH POSSIBLE REMEDIATION
# ==========================================

def evaluate_actions(
    graph,
    model,
    device,
    threshold
):

    original_risk = measure_risk(
        graph,
        model,
        device,
        threshold
    )

    actions = generate_actions(graph)

    results = []

    for action in actions:

        modified = apply_intervention(
            graph,
            action
        )

        after = measure_risk(
            modified,
            model,
            device,
            threshold
        )

        ai_reduction = (
            original_risk["ai_risk_score"]
            - after["ai_risk_score"]
        )

        path_reduction = (
            original_risk["attack_paths"]
            - after["attack_paths"]
        )

        asset_reduction = (
            original_risk[
                "reachable_critical_assets"
            ]
            - after[
                "reachable_critical_assets"
            ]
        )

        cost = action["cost"]

        result = {

            **action,

            "ai_risk_before":
                original_risk["ai_risk_score"],

            "ai_risk_after":
                after["ai_risk_score"],

            "ai_risk_reduction":
                ai_reduction,

            "attack_paths_before":
                original_risk["attack_paths"],

            "attack_paths_after":
                after["attack_paths"],

            "attack_paths_removed":
                path_reduction,

            "critical_assets_before":
                original_risk[
                    "reachable_critical_assets"
                ],

            "critical_assets_after":
                after[
                    "reachable_critical_assets"
                ],

            "critical_assets_protected":
                asset_reduction,

            "high_risk_nodes_reduced":
                original_risk[
                    "simulated_high_risk_nodes"
                ]
                - after[
                    "simulated_high_risk_nodes"
                ],

            "ai_efficiency":
                ai_reduction / cost,

            "path_efficiency":
                path_reduction / cost,

            "asset_efficiency":
                asset_reduction / cost
        }

        results.append(result)

    # AI ranking and simulator ranking are
    # independent and can be compared.

    results.sort(
        key=lambda item: item["ai_efficiency"],
        reverse=True
    )

    return original_risk, results


# ==========================================
# SELECT EXAMPLE ENVIRONMENT
# ==========================================

def select_scenario():

    manifest_path = (
        DATASET_DIR
        / "test"
        / "manifest.json"
    )

    with open(manifest_path) as file:

        manifest = json.load(file)

    # Demonstration scenario only.
    # Select the first graph with multiple paths.

    for filename in manifest["files"]:

        graph = load_graph(
            DATASET_DIR
            / "labeled"
            / filename
        )

        if len(find_attack_paths(graph)) >= 5:

            return filename, graph

    raise RuntimeError(
        "No suitable scenario found"
    )


# ==========================================
# VISUALIZE REMEDIATION EFFECTIVENESS
# ==========================================

def plot_remediation(results):

    top = results[:10]

    if not top:
        return

    names = [

        f"{item['action_type']}\n"
        f"{item.get('source', '')} "
        f"{item['target']}"

        for item in top
    ]

    values = [
        item["ai_efficiency"]
        for item in top
    ]

    plt.figure(figsize=(12, 7))

    plt.barh(
        names[::-1],
        values[::-1]
    )

    plt.xlabel(
        "AI risk-score reduction per cost unit"
    )

    plt.title(
        "Cost-Aware Counterfactual Remediation"
    )

    plt.tight_layout()

    plt.savefig(

        RESULTS_DIR
        / "remediation_ranking.png",

        dpi=300
    )

    plt.close()


# ==========================================
# GRAPH VISUALIZATION
# ==========================================

def draw_graph(graph, filename, title):

    paths = find_attack_paths(graph)

    risky_nodes = {
        node
        for path in paths
        for node in path[1:]
    }

    risky_edges = {
        (source, target)
        for path in paths
        for source, target in zip(
            path[:-1],
            path[1:]
        )
    }

    node_colors = [

        "tomato"
        if node in risky_nodes
        else "lightblue"

        for node in graph.nodes()
    ]

    edge_colors = [

        "red"
        if edge in risky_edges
        else "gray"

        for edge in graph.edges()
    ]

    position = nx.spring_layout(
        graph,
        seed=42
    )

    plt.figure(figsize=(16, 12))

    nx.draw_networkx(

        graph,

        pos=position,

        node_color=node_colors,

        edge_color=edge_colors,

        node_size=850,

        font_size=7,

        arrows=True,

        arrowsize=15
    )

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / filename,
        dpi=300
    )

    plt.close()


# ==========================================
# MAIN PIPELINE
# ==========================================

def main():

    start = time.time()

    print("=" * 65)
    print("GRAPH ML CLOUD SECURITY")
    print("Counterfactual Remediation Engine")
    print("=" * 65)

    model, device, threshold = load_model()

    filename, graph = select_scenario()

    print(f"Scenario: {filename}")
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")
    print(f"Device: {device}")

    original_risk, results = evaluate_actions(
        graph,
        model,
        device,
        threshold
    )

    if not results:

        print("No remediation actions available.")
        return

    print("\nOriginal environment:")

    print(json.dumps(
        original_risk,
        indent=4
    ))

    print(
        f"\nCandidate actions: {len(results)}"
    )

    print("\nTOP 10 AI-RANKED REMEDIATIONS")

    for rank, result in enumerate(
        results[:10],
        start=1
    ):

        print(
            f"\nRank {rank}: "
            f"{result['action_type']}"
        )

        print(
            f"  Target: {result['target']}"
        )

        if "source" in result:

            print(
                f"  Source: {result['source']}"
            )

        print(f"  Cost: {result['cost']}")

        print(
            "  AI risk reduction: "
            f"{result['ai_risk_reduction']:.4f}"
        )

        print(
            "  Attack paths removed: "
            f"{result['attack_paths_removed']}"
        )

        print(
            "  Critical assets protected: "
            f"{result['critical_assets_protected']}"
        )

        print(
            "  AI efficiency: "
            f"{result['ai_efficiency']:.4f}"
        )

    best_ai = results[0]

    best_rule = max(
        results,
        key=lambda item: item["asset_efficiency"]
    )

    severity_actions = [
        item for item in results
        if item["action_type"]
        == "patch_vulnerability"
    ]

    severity_baseline = None

    if severity_actions:

        severity_baseline = max(

            severity_actions,

            key=lambda item:
                graph.nodes[
                    item["target"]
                ]["severity"]
        )

    # Save full candidate ranking

    pd.DataFrame(results).to_csv(

        RESULTS_DIR
        / "remediation_candidates.csv",

        index=False
    )

    report = {

        "project":
            "GraphML-CloudSecurity",

        "experiment":
            "counterfactual_remediation",

        "scenario": filename,

        "source": "synthetic",

        "demonstration_only": True,

        "cost_assumptions": ACTION_COSTS,

        "original_risk": original_risk,

        "candidate_actions": len(results),

        "best_ai_action": best_ai,

        "best_rule_action": best_rule,

        "severity_baseline_action":
            severity_baseline,

        "top_10_actions": results[:10],

        "execution_time_seconds": round(
            time.time() - start,
            2
        )
    }

    with open(

        RESULTS_DIR
        / "remediation_summary.json",

        "w"

    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    plot_remediation(results)

    draw_graph(

        graph,

        "graph_before_remediation.png",

        "Cloud Risk Before Remediation"
    )

    remediated = apply_intervention(
        graph,
        best_ai
    )

    draw_graph(

        remediated,

        "graph_after_remediation.png",

        "Cloud Risk After AI-Ranked Remediation"
    )

    print("\n" + "=" * 65)
    print("REMEDIATION ANALYSIS COMPLETE")
    print("=" * 65)

    print(json.dumps(
        report,
        indent=4
    ))

    print(
        f"\nResults saved to: {RESULTS_DIR}"
    )


if __name__ == "__main__":
    main()
