
import json
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from counterfactual_remediation import (
    load_model,
    load_graph,
    find_attack_paths,
    generate_actions,
    apply_intervention,
    predict_graph_risk
)

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

SEED = 42

# Start with 100 environments to verify everything.
# Change to 3000 after the initial run succeeds.

MAX_SCENARIOS = 100

BUDGETS = [2, 5, 10, 20]

STRATEGIES = [
    "severity",
    "graphsage",
    "graph_rules"
]


# ==========================================
# RISK METRICS
# ==========================================

def simulator_metrics(graph):

    paths = find_attack_paths(graph)

    return {
        "paths": len(paths),

        "assets": len({
            path[-1] for path in paths
        }),

        "risky_nodes": len({
            node
            for path in paths
            for node in path[1:]
        })
    }


def ai_risk(graph, model, device, threshold):

    predictions = predict_graph_risk(
        graph,
        model,
        device,
        threshold
    )

    return sum(
        item["risk_score"]
        for item in predictions.values()
    )


# ==========================================
# IDENTIFY UNIQUE ACTIONS
# ==========================================

def action_key(action):

    return (
        action["action_type"],
        action.get("source", ""),
        action["target"]
    )


# ==========================================
# SELECT NEXT REMEDIATION
# ==========================================

def select_action(
    graph,
    actions,
    strategy,
    model,
    device,
    threshold
):

    if not actions:
        return None

    if strategy == "severity":

        patches = [
            action
            for action in actions
            if action["action_type"]
            == "patch_vulnerability"
        ]

        if not patches:
            return None

        return max(
            patches,
            key=lambda action:
                graph.nodes[
                    action["target"]
                ]["severity"]
        )

    best_action = None
    best_score = 0.0

    if strategy == "graphsage":

        before = ai_risk(
            graph,
            model,
            device,
            threshold
        )

    else:

        before = simulator_metrics(graph)

    for action in actions:

        modified = apply_intervention(
            graph,
            action
        )

        cost = action["cost"]

        if strategy == "graphsage":

            after = ai_risk(
                modified,
                model,
                device,
                threshold
            )

            score = (
                before - after
            ) / cost

        elif strategy == "graph_rules":

            after = simulator_metrics(
                modified
            )

            # Prioritize critical assets.
            # Use path reduction as a
            # secondary ranking criterion.

            assets_removed = (
                before["assets"]
                - after["assets"]
            )

            paths_removed = (
                before["paths"]
                - after["paths"]
            )

            score = (
                assets_removed
                + 0.001 * paths_removed
            ) / cost

        else:

            raise ValueError(
                f"Unknown strategy: {strategy}"
            )

        if score > best_score + 1e-8:

            best_action = action
            best_score = score

    return best_action


# ==========================================
# BUDGET-CONSTRAINED OPTIMIZATION
# ==========================================

def optimize(
    graph,
    strategy,
    budget,
    model,
    device,
    threshold
):

    start = time.time()

    current = graph.copy()

    original = simulator_metrics(current)

    remaining = float(budget)

    applied = []
    used_actions = set()

    while remaining > 0:

        # No modeled risk remains for the
        # exact-rule strategy to reduce.

        if strategy == "graph_rules":

            if simulator_metrics(current)["paths"] == 0:
                break

        candidates = [

            action

            for action in generate_actions(current)

            if action["cost"] <= remaining

            and action_key(action)
                not in used_actions
        ]

        action = select_action(

            current,
            candidates,
            strategy,
            model,
            device,
            threshold
        )

        if action is None:
            break

        current = apply_intervention(
            current,
            action
        )

        remaining -= action["cost"]

        used_actions.add(
            action_key(action)
        )

        applied.append(action)

    final = simulator_metrics(current)

    cost = budget - remaining

    paths_removed = (
        original["paths"] - final["paths"]
    )

    assets_protected = (
        original["assets"] - final["assets"]
    )

    risky_nodes_reduced = (
        original["risky_nodes"]
        - final["risky_nodes"]
    )

    return {

        "strategy": strategy,

        "budget": budget,

        "cost_used": cost,

        "actions_applied": len(applied),

        "paths_before": original["paths"],

        "paths_after": final["paths"],

        "paths_removed": paths_removed,

        "assets_before": original["assets"],

        "assets_after": final["assets"],

        "assets_protected": assets_protected,

        "risky_nodes_reduced":
            risky_nodes_reduced,

        "complete_remediation": int(
            original["paths"] > 0
            and final["paths"] == 0
        ),

        "execution_time_seconds":
            time.time() - start,

        "selected_action_details":
            json.dumps(applied)
    }


# ==========================================
# AGGREGATE EXPERIMENTAL RESULTS
# ==========================================

def aggregate_results(dataframe):

    summaries = []

    for budget in BUDGETS:

        for strategy in STRATEGIES:

            rows = dataframe[
                (dataframe["budget"] == budget)

                &

                (dataframe["strategy"] == strategy)
            ]

            risky = rows[
                rows["paths_before"] > 0
            ]

            paths_before = int(
                rows["paths_before"].sum()
            )

            paths_removed = int(
                rows["paths_removed"].sum()
            )

            assets_before = int(
                rows["assets_before"].sum()
            )

            assets_protected = int(
                rows["assets_protected"].sum()
            )

            total_cost = float(
                rows["cost_used"].sum()
            )

            summaries.append({

                "strategy": strategy,

                "budget": budget,

                "environments": len(rows),

                "risky_environments": len(risky),

                "total_paths_before":
                    paths_before,

                "total_paths_removed":
                    paths_removed,

                "path_reduction_percentage":
                    100 * paths_removed
                    / max(1, paths_before),

                "total_assets_before":
                    assets_before,

                "total_assets_protected":
                    assets_protected,

                "asset_protection_percentage":
                    100 * assets_protected
                    / max(1, assets_before),

                "complete_remediation_percentage":

                    100
                    * risky["complete_remediation"].sum()
                    / max(1, len(risky)),

                "total_cost": total_cost,

                "average_cost_per_environment":

                    total_cost / max(1, len(rows)),

                "assets_protected_per_cost_unit":

                    assets_protected
                    / max(1e-9, total_cost),

                "average_runtime_seconds":

                    float(
                        rows[
                            "execution_time_seconds"
                        ].mean()
                    )
            })

    return pd.DataFrame(summaries)


# ==========================================
# VISUALIZE BUDGET EFFECTIVENESS
# ==========================================

def plot_results(summary):

    names = {
        "severity": "Severity Baseline",
        "graphsage": "GraphSAGE",
        "graph_rules": "Exact Graph Rules"
    }

    metrics = [

        (
            "path_reduction_percentage",
            "Attack Paths Removed (%)",
            "budget_path_reduction.png"
        ),

        (
            "asset_protection_percentage",
            "Critical Assets Protected (%)",
            "budget_asset_protection.png"
        ),

        (
            "complete_remediation_percentage",
            "Complete Remediation (%)",
            "budget_complete_remediation.png"
        ),

        (
            "assets_protected_per_cost_unit",
            "Assets Protected per Cost Unit",
            "budget_cost_efficiency.png"
        )
    ]

    for column, ylabel, filename in metrics:

        plt.figure(figsize=(10, 6))

        for strategy in STRATEGIES:

            rows = summary[
                summary["strategy"] == strategy
            ].sort_values("budget")

            plt.plot(
                rows["budget"],
                rows[column],
                marker="o",
                linewidth=2,
                label=names[strategy]
            )

        plt.xlabel("Remediation Budget (Cost Units)")
        plt.ylabel(ylabel)

        plt.title(
            "Budget-Constrained Cloud Remediation"
        )

        plt.xticks(BUDGETS)
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()

        plt.savefig(
            RESULTS_DIR / filename,
            dpi=300
        )

        plt.close()


# ==========================================
# MAIN EXPERIMENT
# ==========================================

def main():

    start = time.time()

    model, device, threshold = load_model()

    with open(
        DATASET_DIR / "test" / "manifest.json"
    ) as file:

        manifest = json.load(file)

    filenames = sorted(
        manifest["files"]
    )

    random.Random(SEED).shuffle(
        filenames
    )

    filenames = filenames[:MAX_SCENARIOS]

    results = []

    print("=" * 70)
    print("GRAPH ML CLOUD SECURITY")
    print("Large-Scale Budget-Constrained Remediation")
    print("=" * 70)

    print(f"Device: {device}")
    print(f"Environments: {len(filenames)}")
    print(f"Budgets: {BUDGETS}")
    print(f"Strategies: {STRATEGIES}")

    print(
        f"Total experiments: "
        f"{len(filenames) * len(BUDGETS) * len(STRATEGIES):,}"
    )

    for index, filename in enumerate(
        filenames,
        start=1
    ):

        graph = load_graph(

            DATASET_DIR
            / "labeled"
            / filename
        )

        for budget in BUDGETS:

            for strategy in STRATEGIES:

                result = optimize(

                    graph,

                    strategy,

                    budget,

                    model,

                    device,

                    threshold
                )

                result["scenario"] = filename

                results.append(result)

        if index % 10 == 0 or index == len(filenames):

            elapsed = time.time() - start

            print(

                f"Processed "
                f"{index:,}/{len(filenames):,} "
                f"| {elapsed:.1f}s",

                flush=True
            )

            # Save intermediate results.

            pd.DataFrame(results).to_csv(

                RESULTS_DIR
                / "remediation_progress.csv",

                index=False
            )

    dataframe = pd.DataFrame(results)

    dataframe.to_csv(

        RESULTS_DIR
        / "large_scale_remediation.csv",

        index=False
    )

    summary = aggregate_results(
        dataframe
    )

    summary.to_csv(

        RESULTS_DIR
        / "remediation_budget_summary.csv",

        index=False
    )

    plot_results(summary)

    report = {

        "project": "GraphML-CloudSecurity",

        "source": "synthetic",

        "experiment":
            "budget_constrained_remediation",

        "optimization_method":
            "greedy_sequential",

        "random_seed": SEED,

        "environments": len(filenames),

        "budgets": BUDGETS,

        "strategies": STRATEGIES,

        "total_experiments": len(dataframe),

        "execution_time_seconds":

            round(time.time() - start, 2),

        "results":

            summary.to_dict(orient="records")
    }

    with open(

        RESULTS_DIR
        / "large_scale_remediation_summary.json",

        "w"

    ) as file:

        json.dump(report, file, indent=4)

    print("\n" + "=" * 70)
    print("REMEDIATION EVALUATION COMPLETE")
    print("=" * 70)

    print(

        summary[[

            "strategy",

            "budget",

            "path_reduction_percentage",

            "asset_protection_percentage",

            "complete_remediation_percentage",

            "assets_protected_per_cost_unit"

        ]].round(4).to_string(index=False)
    )

    print(
        f"\nExecution time: "
        f"{time.time() - start:.2f} seconds"
    )

    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":

    main()
