
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "datasets" / "processed"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

INPUT_DIM = 8
HIDDEN_DIM = 64
OUTPUT_HIDDEN_DIM = 32

EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
DROPOUT = 0.3
PATIENCE = 10


# ==========================================
# REPRODUCIBILITY
# ==========================================

def set_seed():

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


# ==========================================
# DEVICE SELECTION
# ==========================================

def select_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    # CPU is the safer initial choice for
    # PyTorch Geometric compatibility on Mac.

    return torch.device("cpu")


# ==========================================
# GRAPHSAGE MODEL
# ==========================================

class GraphSAGE(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = SAGEConv(
            INPUT_DIM,
            HIDDEN_DIM
        )

        self.conv2 = SAGEConv(
            HIDDEN_DIM,
            HIDDEN_DIM
        )

        self.conv3 = SAGEConv(
            HIDDEN_DIM,
            OUTPUT_HIDDEN_DIM
        )

        self.classifier = nn.Linear(
            OUTPUT_HIDDEN_DIM,
            1
        )

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = F.dropout(
            x,
            p=DROPOUT,
            training=self.training
        )

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = F.dropout(
            x,
            p=DROPOUT,
            training=self.training
        )

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        x = self.classifier(x)

        return x.squeeze(-1)


# ==========================================
# LOAD DATASETS
# ==========================================

def load_datasets():

    print("Loading training dataset...")

    # Only load trusted, locally generated files.
    train_data = torch.load(
        DATASET_DIR / "train.pt",
        map_location="cpu",
        weights_only=False
    )

    print("Loading validation dataset...")

    validation_data = torch.load(
        DATASET_DIR / "validation.pt",
        map_location="cpu",
        weights_only=False
    )

    print("Loading dataset configuration...")

    with open(
        DATASET_DIR / "processed_summary.json",
        "r",
        encoding="utf-8"
    ) as file:

        summary = json.load(file)

    return train_data, validation_data, summary


# ==========================================
# EVALUATION METRICS
# ==========================================

def calculate_metrics(
    labels,
    probabilities,
    threshold=0.5
):

    labels = np.asarray(labels)

    probabilities = np.asarray(
        probabilities
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {

        "accuracy": float(
            accuracy_score(
                labels, predictions
            )
        ),

        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                labels,
                predictions,
                zero_division=0
            )
        ),

        "f1": float(
            f1_score(
                labels,
                predictions,
                zero_division=0
            )
        ),

        "average_precision": float(
            average_precision_score(
                labels, probabilities
            )
        ),

        "threshold": threshold,

        "confusion_matrix": confusion_matrix(
            labels,
            predictions,
            labels=[0, 1]
        ).tolist()
    }

    if len(np.unique(labels)) == 2:

        metrics["roc_auc"] = float(
            roc_auc_score(
                labels, probabilities
            )
        )

    else:

        metrics["roc_auc"] = None

    return metrics


# ==========================================
# TRAIN ONE EPOCH
# ==========================================

def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device
):

    model.train()

    total_loss = 0.0
    total_nodes = 0

    for batch in loader:

        batch = batch.to(device)

        optimizer.zero_grad()

        logits = model(
            batch.x,
            batch.edge_index
        )

        mask = batch.supervised_mask

        labels = batch.y[mask].float()

        loss = criterion(
            logits[mask],
            labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer.step()

        nodes = int(mask.sum().item())

        total_loss += loss.item() * nodes
        total_nodes += nodes

    return total_loss / max(1, total_nodes)


# ==========================================
# VALIDATE MODEL
# ==========================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    threshold=0.5
):

    model.eval()

    total_loss = 0.0
    total_nodes = 0

    all_labels = []
    all_probabilities = []

    for batch in loader:

        batch = batch.to(device)

        logits = model(
            batch.x,
            batch.edge_index
        )

        mask = batch.supervised_mask

        labels = batch.y[mask].float()

        selected_logits = logits[mask]

        loss = criterion(
            selected_logits,
            labels
        )

        probabilities = torch.sigmoid(
            selected_logits
        )

        count = int(mask.sum().item())

        total_loss += loss.item() * count
        total_nodes += count

        all_labels.extend(
            labels.cpu().numpy().tolist()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy().tolist()
        )

    metrics = calculate_metrics(
        all_labels,
        all_probabilities,
        threshold
    )

    metrics["loss"] = (
        total_loss / max(1, total_nodes)
    )

    return (
        metrics,
        np.asarray(all_labels),
        np.asarray(all_probabilities)
    )


# ==========================================
# SELECT VALIDATION THRESHOLD
# ==========================================

def find_best_threshold(
    labels,
    probabilities
):

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in np.linspace(
        0.05, 0.95, 91
    ):

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            labels,
            predictions,
            zero_division=0
        )

        if score > best_f1:

            best_f1 = score

            best_threshold = float(
                threshold
            )

    return round(best_threshold, 4)


# ==========================================
# MAIN TRAINING PIPELINE
# ==========================================

def main():

    set_seed()

    start_time = time.time()

    device = select_device()

    print("=" * 70)
    print("GRAPH ML CLOUD SECURITY")
    print("GraphSAGE Model Training")
    print("=" * 70)

    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Maximum epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("=" * 70)

    # Load training and validation data

    train_data, validation_data, summary = (
        load_datasets()
    )

    if summary["number_of_features"] != INPUT_DIM:

        raise ValueError(
            "Unexpected number of node features"
        )

    # Data loaders

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(

        train_data,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=0,

        generator=generator
    )

    validation_loader = DataLoader(

        validation_data,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0
    )

    print(f"Training graphs: {len(train_data):,}")

    print(
        f"Validation graphs: "
        f"{len(validation_data):,}"
    )

    # Address class imbalance

    positive_weight = summary[
        "training_positive_class_weight"
    ]

    print(
        f"Positive class weight: "
        f"{positive_weight}"
    )

    criterion = nn.BCEWithLogitsLoss(

        pos_weight=torch.tensor(

            [positive_weight],

            dtype=torch.float32,

            device=device
        )
    )

    # Initialize model

    model = GraphSAGE().to(device)

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(

            optimizer,

            mode="max",

            factor=0.5,

            patience=4
        )
    )

    parameter_count = sum(

        parameter.numel()

        for parameter in model.parameters()

        if parameter.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{parameter_count:,}"
    )

    print("\nStarting model training...\n")

    best_ap = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    history = []

    model_path = (
        MODEL_DIR / "best_graphsage.pt"
    )

    # ======================================
    # TRAINING LOOP
    # ======================================

    for epoch in range(1, EPOCHS + 1):

        epoch_start = time.time()

        train_loss = train_epoch(

            model,

            train_loader,

            optimizer,

            criterion,

            device
        )

        validation_metrics, _, _ = evaluate(

            model,

            validation_loader,

            criterion,

            device
        )

        validation_ap = validation_metrics[
            "average_precision"
        ]

        scheduler.step(validation_ap)

        epoch_time = (
            time.time() - epoch_start
        )

        record = {

            "epoch": epoch,

            "train_loss": train_loss,

            "validation_loss":
                validation_metrics["loss"],

            "validation_accuracy":
                validation_metrics["accuracy"],

            "validation_precision":
                validation_metrics["precision"],

            "validation_recall":
                validation_metrics["recall"],

            "validation_f1":
                validation_metrics["f1"],

            "validation_roc_auc":
                validation_metrics["roc_auc"],

            "validation_average_precision":
                validation_ap,

            "learning_rate":
                optimizer.param_groups[0]["lr"],

            "epoch_time_seconds": epoch_time
        }

        history.append(record)

        print(

            f"Epoch {epoch:02d}/{EPOCHS} | "

            f"Loss: {train_loss:.4f} | "

            f"Val Loss: "
            f"{validation_metrics['loss']:.4f} | "

            f"Precision: "
            f"{validation_metrics['precision']:.4f} | "

            f"Recall: "
            f"{validation_metrics['recall']:.4f} | "

            f"F1: "
            f"{validation_metrics['f1']:.4f} | "

            f"AP: {validation_ap:.4f} | "

            f"Time: {epoch_time:.1f}s"
        )

        # Save training history after each epoch

        with open(

            RESULTS_DIR / "training_history.json",

            "w",

            encoding="utf-8"

        ) as file:

            json.dump(history, file, indent=4)

        # Save best model using validation AP

        if validation_ap > best_ap + 0.00001:

            best_ap = validation_ap
            best_epoch = epoch

            epochs_without_improvement = 0

            torch.save(

                {
                    "model_state_dict":
                        model.state_dict(),

                    "epoch": epoch,

                    "validation_ap": best_ap,

                    "input_dim": INPUT_DIM,

                    "hidden_dim": HIDDEN_DIM,

                    "output_hidden_dim":
                        OUTPUT_HIDDEN_DIM,

                    "dropout": DROPOUT
                },

                model_path
            )

            print("  Saved new best model.")

        else:

            epochs_without_improvement += 1

        # Early stopping

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(

                f"\nEarly stopping at "
                f"epoch {epoch}"

            )

            break

    # ======================================
    # LOAD BEST MODEL
    # ======================================

    checkpoint = torch.load(

        model_path,

        map_location=device,

        weights_only=True
    )

    model.load_state_dict(

        checkpoint["model_state_dict"]
    )

    # Optimize threshold on validation only

    _, labels, probabilities = evaluate(

        model,

        validation_loader,

        criterion,

        device
    )

    best_threshold = find_best_threshold(

        labels,

        probabilities
    )

    final_validation_metrics = (

        calculate_metrics(

            labels,

            probabilities,

            threshold=best_threshold
        )
    )

    checkpoint["decision_threshold"] = (
        best_threshold
    )

    torch.save(

        checkpoint,

        model_path
    )

    # ======================================
    # FINAL TRAINING SUMMARY
    # ======================================

    training_summary = {

        "project": "GraphML-CloudSecurity",

        "model": "GraphSAGE",

        "device": str(device),

        "random_seed": SEED,

        "input_features": INPUT_DIM,

        "hidden_dimensions": [
            HIDDEN_DIM,
            HIDDEN_DIM,
            OUTPUT_HIDDEN_DIM
        ],

        "trainable_parameters": parameter_count,

        "training_graphs": len(train_data),

        "validation_graphs":
            len(validation_data),

        "maximum_epochs": EPOCHS,

        "completed_epochs": len(history),

        "best_epoch": best_epoch,

        "batch_size": BATCH_SIZE,

        "initial_learning_rate":
            LEARNING_RATE,

        "positive_class_weight":
            positive_weight,

        "decision_threshold":
            best_threshold,

        "validation_metrics":
            final_validation_metrics,

        "test_set_used": False,

        "training_time_seconds": round(

            time.time() - start_time,

            2
        ),

        "model_path": str(model_path)
    }

    summary_path = (

        RESULTS_DIR / "training_summary.json"
    )

    with open(

        summary_path,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            training_summary,

            file,

            indent=4
        )

    print("\n" + "=" * 70)
    print("GRAPHSAGE TRAINING COMPLETE")
    print("=" * 70)

    print(

        json.dumps(

            training_summary,

            indent=4
        )
    )

    print("=" * 70)

    print(f"Best model: {model_path}")

    print(
        f"Training summary: {summary_path}"
    )

    print("=" * 70)


if __name__ == "__main__":

    main()
