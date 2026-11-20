
import json
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from pathlib import Path
from torch_geometric.loader import DataLoader

from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    roc_auc_score,
    ConfusionMatrixDisplay
)

from train_graphsage import (
    GraphSAGE,
    calculate_metrics,
    select_device
)

# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "datasets" / "processed"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 64

# ==========================================
# LOAD TRAINED MODEL
# ==========================================

def load_model(device):

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

    threshold = checkpoint["decision_threshold"]

    return model, checkpoint, threshold


# ==========================================
# EVALUATE UNSEEN TEST ENVIRONMENTS
# ==========================================

@torch.no_grad()
def predict_test_data(model, loader, device):

    all_labels = []
    all_scores = []
    all_severity = []
    all_node_types = []

    for batch in loader:

        batch = batch.to(device)

        logits = model(
            batch.x,
            batch.edge_index
        )

        mask = batch.supervised_mask

        scores = torch.sigmoid(logits[mask])

        all_scores.extend(
            scores.cpu().tolist()
        )

        all_labels.extend(
            batch.y[mask].cpu().tolist()
        )

        # Feature index 4 is normalized severity.
        all_severity.extend(
            batch.x[mask, 4].cpu().tolist()
        )

        all_node_types.extend(
            batch.node_type[mask].cpu().tolist()
        )

    return (
        np.asarray(all_labels),
        np.asarray(all_scores),
        np.asarray(all_severity),
        np.asarray(all_node_types)
    )


# ==========================================
# PLOT TRAINING HISTORY
# ==========================================

def plot_training_history():

    with open(
        RESULTS_DIR / "training_history.json",
        "r",
        encoding="utf-8"
    ) as file:

        history = json.load(file)

    epochs = [
        row["epoch"] for row in history
    ]

    train_loss = [
        row["train_loss"] for row in history
    ]

    val_loss = [
        row["validation_loss"] for row in history
    ]

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        train_loss,
        label="Training Loss",
        linewidth=2
    )

    plt.plot(
        epochs,
        val_loss,
        label="Validation Loss",
        linewidth=2
    )

    plt.xlabel("Epoch")
    plt.ylabel("Weighted BCE Loss")
    plt.title("GraphSAGE Training Convergence")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "training_loss.png",
        dpi=300
    )

    plt.close()

    plt.figure(figsize=(10, 6))

    metrics = {
        "validation_precision": "Precision",
        "validation_recall": "Recall",
        "validation_f1": "F1 Score",
        "validation_average_precision":
            "Average Precision"
    }

    for key, name in metrics.items():

        values = [
            row[key] for row in history
        ]

        plt.plot(
            epochs,
            values,
            label=name,
            linewidth=2
        )

    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("GraphSAGE Validation Performance")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "training_metrics.png",
        dpi=300
    )

    plt.close()


# ==========================================
# PLOT TEST RESULTS
# ==========================================

def plot_test_results(
    labels,
    scores,
    severity,
    threshold
):

    predictions = (
        scores >= threshold
    ).astype(int)

    # Confusion matrix

    ConfusionMatrixDisplay.from_predictions(
        labels,
        predictions,
        labels=[0, 1],
        display_labels=[
            "Negative",
            "High Risk"
        ],
        cmap="Blues",
        values_format="d"
    )

    plt.title("GraphSAGE Test Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "test_confusion_matrix.png",
        dpi=300
    )

    plt.close()

    # ROC curve

    plt.figure(figsize=(8, 6))

    for values, name in [
        (scores, "GraphSAGE"),
        (severity, "Severity Baseline")
    ]:

        fpr, tpr, _ = roc_curve(
            labels,
            values
        )

        auc = roc_auc_score(
            labels,
            values
        )

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{name} (AUC={auc:.4f})"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Baseline"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Test ROC Curve")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "test_roc_curve.png",
        dpi=300
    )

    plt.close()

    # Precision-recall curve

    plt.figure(figsize=(8, 6))

    for values, name in [
        (scores, "GraphSAGE"),
        (severity, "Severity Baseline")
    ]:

        precision, recall, _ = (
            precision_recall_curve(
                labels,
                values
            )
        )

        ap = average_precision_score(
            labels,
            values
        )

        plt.plot(
            recall,
            precision,
            linewidth=2,
            label=f"{name} (AP={ap:.4f})"
        )

    plt.axhline(
        labels.mean(),
        linestyle="--",
        label="Positive Class Prevalence"
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Test Precision-Recall Comparison")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "test_precision_recall.png",
        dpi=300
    )

    plt.close()


# ==========================================
# MAIN EVALUATION PIPELINE
# ==========================================

def main():

    start_time = time.time()

    device = select_device()

    print("=" * 65)
    print("GRAPH ML CLOUD SECURITY")
    print("Independent Test Evaluation")
    print("=" * 65)

    # Load only our locally generated test data.

    test_data = torch.load(
        DATASET_DIR / "test.pt",
        map_location="cpu",
        weights_only=False
    )

    loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model, checkpoint, threshold = load_model(
        device
    )

    print(f"Device: {device}")
    print(f"Test environments: {len(test_data):,}")
    print(f"Best epoch: {checkpoint['epoch']}")
    print(f"Frozen threshold: {threshold}")

    inference_start = time.time()

    labels, scores, severity, types = (
        predict_test_data(
            model,
            loader,
            device
        )
    )

    inference_time = (
        time.time() - inference_start
    )

    # GraphSAGE evaluation

    graphsage_metrics = calculate_metrics(
        labels,
        scores,
        threshold=threshold
    )

    # Traditional severity-only baseline.
    # Normalized severity >= 0.7 corresponds
    # to a synthetic severity score >= 7.0.

    severity_metrics = calculate_metrics(
        labels,
        severity,
        threshold=0.7
    )

    # Majority-class baseline

    majority_metrics = calculate_metrics(
        labels,
        np.zeros_like(scores),
        threshold=0.5
    )

    # ======================================
    # PERFORMANCE BY RESOURCE TYPE
    # ======================================

    resource_metrics = {}

    resource_types = {
        1: "EC2",
        2: "IAM",
        3: "S3"
    }

    for type_id, name in resource_types.items():

        mask = types == type_id

        resource_metrics[name] = {
            "total_resources": int(mask.sum()),
            "positive_resources": int(
                labels[mask].sum()
            ),
            "metrics": calculate_metrics(
                labels[mask],
                scores[mask],
                threshold=threshold
            )
        }

    # EC2-only comparison provides a more
    # meaningful severity baseline because
    # IAM and S3 have zero synthetic severity.

    ec2_mask = types == 1

    ec2_comparison = {

        "graphsage": calculate_metrics(
            labels[ec2_mask],
            scores[ec2_mask],
            threshold=threshold
        ),

        "severity_baseline": calculate_metrics(
            labels[ec2_mask],
            severity[ec2_mask],
            threshold=0.7
        )
    }

    # ======================================
    # GENERATE RESEARCH FIGURES
    # ======================================

    plot_training_history()

    plot_test_results(
        labels,
        scores,
        severity,
        threshold
    )

    # ======================================
    # SAVE EVALUATION SUMMARY
    # ======================================

    summary = {

        "project": "GraphML-CloudSecurity",

        "evaluation": "independent_test",

        "source": "synthetic",

        "test_graphs": len(test_data),

        "supervised_test_nodes": int(
            len(labels)
        ),

        "positive_test_nodes": int(
            labels.sum()
        ),

        "negative_test_nodes": int(
            len(labels) - labels.sum()
        ),

        "model_epoch": checkpoint["epoch"],

        "decision_threshold": threshold,

        "threshold_selected_using":
            "validation_dataset",

        "graphsage_metrics":
            graphsage_metrics,

        "severity_baseline_metrics":
            severity_metrics,

        "majority_baseline_metrics":
            majority_metrics,

        "resource_type_metrics":
            resource_metrics,

        "ec2_only_comparison":
            ec2_comparison,

        "f1_improvement_over_severity":
            graphsage_metrics["f1"]
            - severity_metrics["f1"],

        "ap_improvement_over_severity":
            graphsage_metrics["average_precision"]
            - severity_metrics["average_precision"],

        "inference_time_seconds": round(
            inference_time,
            4
        ),

        "resources_per_second": round(
            len(labels) / max(inference_time, 1e-9),
            2
        ),

        "evaluation_time_seconds": round(
            time.time() - start_time,
            2
        )
    }

    output_path = (
        RESULTS_DIR / "test_evaluation.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    print("\n" + "=" * 65)
    print("BASELINE COMPARISON")
    print("=" * 65)

    for name, metrics in [

        ("GraphSAGE", graphsage_metrics),

        ("Severity Baseline", severity_metrics),

        ("Majority Baseline", majority_metrics)

    ]:

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

    print("\n" + "=" * 65)
    print("INDEPENDENT TEST EVALUATION COMPLETE")
    print("=" * 65)

    print(json.dumps(summary, indent=4))

    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
