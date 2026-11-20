
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from torch_geometric.loader import DataLoader

from train_graphsage import (
    GraphSAGE,
    calculate_metrics,
    find_best_threshold,
    select_device
)

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets" / "processed"
RESULTS_DIR = BASE_DIR / "results"
MODEL_DIR = BASE_DIR / "models"

RESULTS_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

SEED = 42
EPOCHS = 50
BATCH_SIZE = 64
PATIENCE = 10
LEARNING_RATE = 0.001


def set_seed():

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def load_split(name):

    return torch.load(
        DATASET_DIR / f"{name}.pt",
        map_location="cpu",
        weights_only=False
    )


# ==========================================
# FEATURE-ONLY NEURAL NETWORK
# ==========================================

class FeatureMLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x, edge_index=None):

        return self.network(x).squeeze(-1)


# ==========================================
# EXTRACT FEATURES FOR LOGISTIC REGRESSION
# ==========================================

def extract_features(dataset):

    features = []
    labels = []

    for graph in dataset:

        mask = graph.supervised_mask

        features.append(
            graph.x[mask].numpy()
        )

        labels.append(
            graph.y[mask].numpy()
        )

    return (
        np.concatenate(features),
        np.concatenate(labels)
    )


# ==========================================
# RANDOMIZE MESSAGE-PASSING RELATIONSHIPS
# ==========================================

def randomize_edges(dataset, seed):

    generator = torch.Generator()
    generator.manual_seed(seed)

    randomized = []

    for graph in dataset:

        modified = graph.clone()

        # Randomly reassign the graph structure
        # to resource nodes within each environment.
        # Preserves graph size and degree distribution.

        permutation = torch.randperm(
            modified.num_nodes,
            generator=generator
        )

        modified.edge_index = permutation[
            modified.edge_index
        ]

        randomized.append(modified)

    return randomized


# ==========================================
# MODEL PREDICTIONS
# ==========================================

@torch.no_grad()
def predict(model, loader, device):

    model.eval()

    labels = []
    scores = []

    for batch in loader:

        batch = batch.to(device)

        logits = model(
            batch.x,
            batch.edge_index
        )

        mask = batch.supervised_mask

        labels.append(
            batch.y[mask].cpu().numpy()
        )

        scores.append(
            torch.sigmoid(
                logits[mask]
            ).cpu().numpy()
        )

    return (
        np.concatenate(labels),
        np.concatenate(scores)
    )


# ==========================================
# TRAIN NEURAL NETWORK EXPERIMENT
# ==========================================

def train_model(
    name,
    model,
    train_data,
    validation_data,
    device,
    positive_weight
):

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0
    )

    validation_loader = DataLoader(
        validation_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0001
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [positive_weight],
            device=device,
            dtype=torch.float32
        )
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=4
        )
    )

    best_ap = -1.0
    best_state = None
    best_epoch = 0
    wait = 0
    start = time.time()

    print("\n" + "=" * 65)
    print(f"Training {name}")
    print("=" * 65)

    for epoch in range(1, EPOCHS + 1):

        model.train()

        total_loss = 0.0
        total_nodes = 0

        for batch in train_loader:

            batch = batch.to(device)

            optimizer.zero_grad()

            logits = model(
                batch.x,
                batch.edge_index
            )

            mask = batch.supervised_mask

            loss = criterion(
                logits[mask],
                batch.y[mask].float()
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0
            )

            optimizer.step()

            count = int(mask.sum().item())

            total_loss += loss.item() * count
            total_nodes += count

        labels, scores = predict(
            model,
            validation_loader,
            device
        )

        metrics = calculate_metrics(
            labels,
            scores
        )

        ap = metrics["average_precision"]

        scheduler.step(ap)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Loss: {total_loss / total_nodes:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"AP: {ap:.4f}"
        )

        if ap > best_ap + 0.00001:

            best_ap = ap
            best_epoch = epoch

            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

            wait = 0

        else:

            wait += 1

        if wait >= PATIENCE:

            print("Early stopping.")
            break

    model.load_state_dict(best_state)

    labels, scores = predict(
        model,
        validation_loader,
        device
    )

    threshold = find_best_threshold(
        labels,
        scores
    )

    validation_metrics = calculate_metrics(
        labels,
        scores,
        threshold=threshold
    )

    checkpoint = {
        "model_state_dict": best_state,
        "decision_threshold": threshold,
        "best_epoch": best_epoch
    }

    filename = (
        name.lower().replace(" ", "_")
        + ".pt"
    )

    torch.save(
        checkpoint,
        MODEL_DIR / filename
    )

    return model, {
        "best_epoch": best_epoch,
        "threshold": threshold,
        "validation_metrics": validation_metrics,
        "training_time_seconds": round(
            time.time() - start, 2
        )
    }


# ==========================================
# EXACT SIMULATOR RULE BASELINE
# ==========================================

def exact_graph_baseline(dataset):

    labels = []
    scores = []

    for graph in dataset:

        prediction = np.zeros(
            graph.num_nodes,
            dtype=int
        )

        node_types = graph.node_type.tolist()
        features = graph.x.numpy()

        adjacency = [
            [] for _ in range(graph.num_nodes)
        ]

        for source, target in (
            graph.directed_edge_index.t().tolist()
        ):

            adjacency[source].append(target)

        for internet in range(graph.num_nodes):

            if node_types[internet] != 0:
                continue

            for ec2 in adjacency[internet]:

                if node_types[ec2] != 1:
                    continue

                if features[ec2, 5] != 1:
                    continue

                if features[ec2, 4] < np.float32(0.7):
                    continue

                for role in adjacency[ec2]:

                    if node_types[role] != 2:
                        continue

                    for bucket in adjacency[role]:

                        if node_types[bucket] != 3:
                            continue

                        if features[bucket, 7] != 1:
                            continue

                        prediction[ec2] = 1
                        prediction[role] = 1
                        prediction[bucket] = 1

        mask = graph.supervised_mask.numpy()

        labels.extend(graph.y.numpy()[mask])
        scores.extend(prediction[mask])

    return calculate_metrics(
        labels,
        scores,
        threshold=0.5
    )


# ==========================================
# VISUALIZE MODEL COMPARISONS
# ==========================================

def plot_results(results):

    names = list(results.keys())

    metrics = [
        "precision",
        "recall",
        "f1",
        "average_precision"
    ]

    positions = np.arange(len(names))
    width = 0.18

    plt.figure(figsize=(13, 7))

    for index, metric in enumerate(metrics):

        values = [
            results[name]["test_metrics"][metric]
            for name in names
        ]

        plt.bar(
            positions + (index - 1.5) * width,
            values,
            width,
            label=metric.replace("_", " ").title()
        )

    plt.xticks(
        positions,
        names,
        rotation=15,
        ha="right"
    )

    plt.ylabel("Score")
    plt.title("Cloud Risk Prediction: Ablation Study")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "ablation_comparison.png",
        dpi=300
    )

    plt.close()


# ==========================================
# MAIN EXPERIMENT
# ==========================================

def main():

    set_seed()

    start = time.time()
    device = select_device()

    print("GRAPH ML CLOUD SECURITY")
    print("Baseline and Ablation Study")
    print(f"Device: {device}")

    train_data = load_split("train")
    validation_data = load_split("validation")
    test_data = load_split("test")

    with open(
        DATASET_DIR / "processed_summary.json"
    ) as file:

        summary = json.load(file)

    weight = summary[
        "training_positive_class_weight"
    ]

    results = {}

    # --------------------------------------
    # EXPERIMENT A: LOGISTIC REGRESSION
    # --------------------------------------

    print("\nTraining logistic regression...")

    x_train, y_train = extract_features(
        train_data
    )

    x_val, y_val = extract_features(
        validation_data
    )

    x_test, y_test = extract_features(
        test_data
    )

    logistic = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=SEED
    )

    logistic.fit(x_train, y_train)

    val_scores = logistic.predict_proba(
        x_val
    )[:, 1]

    threshold = find_best_threshold(
        y_val,
        val_scores
    )

    test_scores = logistic.predict_proba(
        x_test
    )[:, 1]

    results["Logistic Regression"] = {
        "threshold": threshold,
        "validation_metrics": calculate_metrics(
            y_val,
            val_scores,
            threshold
        ),
        "test_metrics": calculate_metrics(
            y_test,
            test_scores,
            threshold
        )
    }

    # --------------------------------------
    # EXPERIMENT B: FEATURE-ONLY MLP
    # --------------------------------------

    set_seed()

    mlp, mlp_results = train_model(
        "Feature MLP",
        FeatureMLP(),
        train_data,
        validation_data,
        device,
        weight
    )

    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    labels, scores = predict(
        mlp,
        test_loader,
        device
    )

    mlp_results["test_metrics"] = (
        calculate_metrics(
            labels,
            scores,
            mlp_results["threshold"]
        )
    )

    results["Feature MLP"] = mlp_results

    # --------------------------------------
    # EXPERIMENT C: RANDOMIZED GRAPH EDGES
    # --------------------------------------

    print("\nRandomizing graph relationships...")

    random_train = randomize_edges(
        train_data,
        SEED + 1
    )

    random_validation = randomize_edges(
        validation_data,
        SEED + 2
    )

    random_test = randomize_edges(
        test_data,
        SEED + 3
    )

    set_seed()

    randomized_model, random_results = (
        train_model(
            "Randomized GraphSAGE",
            GraphSAGE(),
            random_train,
            random_validation,
            device,
            weight
        )
    )

    random_test_loader = DataLoader(
        random_test,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    labels, scores = predict(
        randomized_model,
        random_test_loader,
        device
    )

    random_results["test_metrics"] = (
        calculate_metrics(
            labels,
            scores,
            random_results["threshold"]
        )
    )

    results["Randomized GraphSAGE"] = (
        random_results
    )

    # --------------------------------------
    # EXPERIMENT D: ORIGINAL GRAPHSAGE
    # --------------------------------------

    checkpoint = torch.load(
        MODEL_DIR / "best_graphsage.pt",
        map_location=device,
        weights_only=True
    )

    original_model = GraphSAGE().to(device)

    original_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    labels, scores = predict(
        original_model,
        test_loader,
        device
    )

    results["Original GraphSAGE"] = {
        "threshold": checkpoint[
            "decision_threshold"
        ],
        "best_epoch": checkpoint["epoch"],
        "test_metrics": calculate_metrics(
            labels,
            scores,
            checkpoint["decision_threshold"]
        )
    }

    # --------------------------------------
    # EXACT RULE REFERENCE
    # --------------------------------------

    results["Exact Graph Rules"] = {
        "test_metrics": exact_graph_baseline(
            test_data
        )
    }

    # --------------------------------------
    # SAVE RESULTS
    # --------------------------------------

    report = {
        "project": "GraphML-CloudSecurity",
        "experiment": "baseline_and_ablation",
        "source": "synthetic",
        "random_seed": SEED,
        "training_graphs": len(train_data),
        "validation_graphs": len(validation_data),
        "test_graphs": len(test_data),
        "models": results,
        "execution_time_seconds": round(
            time.time() - start, 2
        )
    }

    with open(
        RESULTS_DIR / "ablation_results.json",
        "w"
    ) as file:

        json.dump(report, file, indent=4)

    plot_results(results)

    print("\n" + "=" * 70)
    print("FINAL ABLATION STUDY RESULTS")
    print("=" * 70)

    for name, result in results.items():

        metrics = result["test_metrics"]

        print(f"\n{name}")

        for metric in [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "average_precision",
            "roc_auc"
        ]:

            value = metrics[metric]

            if value is not None:
                print(f"  {metric}: {value:.6f}")

    print("\nResults saved to:")
    print(RESULTS_DIR / "ablation_results.json")
    print(RESULTS_DIR / "ablation_comparison.png")


if __name__ == "__main__":
    main()
