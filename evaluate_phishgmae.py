import os
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.nn.functional import softmax
from sklearn.metrics import roc_auc_score

# Configuration parameters
NUM_PARTS = 5
DATASET_NAME = "phishscope"
PER_VALUES = ["1", "2", "4"]  # Use three different training ratios

# Detect device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Define MLP model to replace LogReg
class MLP(nn.Module):
    def __init__(
        self, in_dim, out_dim, hidden_dims=[128, 64], dropout=0.2, activation="relu"
    ):
        super(MLP, self).__init__()

        # Define activation function
        if activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "leaky_relu":
            act_fn = nn.LeakyReLU(0.1)
        elif activation == "elu":
            act_fn = nn.ELU()
        else:
            act_fn = nn.ReLU()

        # Build layers
        layers = []

        # Input layer to first hidden layer
        layers.append(nn.Linear(in_dim, hidden_dims[0]))
        layers.append(act_fn)
        layers.append(
            nn.BatchNorm1d(hidden_dims[0])
        )  # Batch normalization improves stability
        layers.append(nn.Dropout(dropout))

        # Additional hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(act_fn)
            layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], out_dim))

        # Combine all layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def encode_onehot(labels):
    """Convert labels to one-hot encoding"""
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot


def evaluate(
    embeds,
    idx_train,
    idx_val,
    idx_test,
    label,
    nb_classes,
    device,
    model_type="mlp",  # New parameter, options: "mlp" or "logreg"
    lr=0.01,
    wd=0.0001,  # Increase weight decay to improve generalization
    hidden_dims=[256, 128],  # Hidden layer dimensions for MLP
    dropout=0.3,  # Dropout rate
    patience=20,  # Early stopping patience
    max_iters=200,
    isTest=True,
    per_value=None,  # Add per_value parameter for file naming when saving
):
    """Evaluate classification performance of embeddings using MLP or LogReg"""
    hid_units = embeds.shape[1]
    xent = nn.CrossEntropyLoss()

    train_embs = embeds[idx_train]
    val_embs = embeds[idx_val]
    test_embs = embeds[idx_test]

    train_lbls = torch.argmax(label[idx_train], dim=-1)
    val_lbls = torch.argmax(label[idx_val], dim=-1)
    test_lbls = torch.argmax(label[idx_test], dim=-1)

    # Store evaluation metrics
    accs = []  # Accuracy
    precisions = []  # Precision
    recalls = []  # Recall
    f1s = []  # F1 score
    auc_score_list = []  # AUC

    # Store predictions of best model
    best_run_predictions = None
    best_run_probabilities = None
    best_run_accuracy = -1

    for i in range(50):
        # Choose model type
        if model_type == "mlp":
            print(f"Run {i+1}/50: Using MLP model") if i == 0 else None
            model = MLP(hid_units, nb_classes, hidden_dims=hidden_dims, dropout=dropout)
        else:  # logreg
            print(f"Run {i+1}/50: Using LogReg model") if i == 0 else None
            # Assuming there's a LogReg class, import or define if needed
            from utils.logreg import LogReg

            model = LogReg(hid_units, nb_classes)

        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        model.to(device)

        val_accs = []
        test_accs = []
        test_precisions = []
        test_recalls = []
        test_f1s = []
        logits_list = []

        # Early stopping counter
        no_improve = 0
        best_val_acc = 0

        for iter_ in range(max_iters):
            # Training phase
            model.train()
            opt.zero_grad()

            logits = model(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

            # Validation phase
            model.eval()
            with torch.no_grad():
                logits = model(val_embs)
            preds = torch.argmax(logits, dim=1)

            val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
            val_accs.append(val_acc.item())

            # Testing phase
            with torch.no_grad():
                logits = model(test_embs)
            preds = torch.argmax(logits, dim=1)

            test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            test_accs.append(test_acc.item())

            # Calculate precision, recall, f1
            test_precision = precision_score(
                test_lbls.cpu(), preds.cpu(), average="macro", zero_division=0
            )
            test_recall = recall_score(
                test_lbls.cpu(), preds.cpu(), average="macro", zero_division=0
            )
            test_f1 = f1_score(
                test_lbls.cpu(), preds.cpu(), average="macro", zero_division=0
            )

            test_precisions.append(test_precision)
            test_recalls.append(test_recall)
            test_f1s.append(test_f1)

            logits_list.append(logits)

            # Early stopping logic
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"Early stopping at iteration {iter_}") if i == 0 else None
                break

        # Select best model iteration based on validation accuracy
        max_iter = val_accs.index(max(val_accs))
        accs.append(test_accs[max_iter])
        precisions.append(test_precisions[max_iter])
        recalls.append(test_recalls[max_iter])
        f1s.append(test_f1s[max_iter])

        # Calculate AUC
        best_logits = logits_list[max_iter]
        best_proba = softmax(best_logits, dim=1)

        y_true = test_lbls.detach().cpu().numpy()

        if nb_classes <= 2:
            y_score = best_proba[:, 1].detach().cpu().numpy()
            auc_score = roc_auc_score(y_true=y_true, y_score=y_score)
        else:
            auc_score = roc_auc_score(
                y_true=test_lbls.detach().cpu().numpy(),
                y_score=best_proba.detach().cpu().numpy(),
                multi_class="ovr",
            )

        auc_score_list.append(auc_score)

        # Save prediction results of best performing model in current run
        if test_accs[max_iter] > best_run_accuracy:
            best_run_accuracy = test_accs[max_iter]
            best_run_predictions = (
                torch.argmax(best_logits, dim=1).detach().cpu().numpy()
            )
            best_run_probabilities = best_proba.detach().cpu().numpy()

    # Save best run prediction results to file
    if isTest and best_run_predictions is not None:
        # Get true labels
        true_labels = test_lbls.detach().cpu().numpy()

        # Create unique filename for different training ratios and model types
        result_filepath = f"./results/phishscope_{per_value}0_{model_type}.txt"

        print(f"\nSaving prediction results to: {result_filepath}")

        with open(result_filepath, "w") as f:
            # Write header
            f.write("# True label, Predicted label, Predicted Probability\n")

            # For binary classification, save probability of class 1; for multi-class, save probability of predicted class
            if nb_classes <= 2:
                for idx in range(len(true_labels)):
                    # Format: true_label predicted_label predicted_probability
                    pred_prob = best_run_probabilities[idx, 1]  # Probability of class 1
                    f.write(
                        f"{true_labels[idx]:.6f}\t{best_run_predictions[idx]:.6f}\t{pred_prob:.6f}\n"
                    )
            else:
                for idx in range(len(true_labels)):
                    # Format: true_label predicted_label predicted_probability
                    pred_class = best_run_predictions[idx]
                    pred_prob = best_run_probabilities[
                        idx, pred_class
                    ]  # Probability of predicted class
                    f.write(
                        f"{true_labels[idx]:.6f}\t{pred_class:.6f}\t{pred_prob:.6f}\n"
                    )

        print(f"Successfully saved {len(true_labels)} URL classification information")

    if isTest:
        print(
            "\t[Classification] Accuracy: [{:.4f}, {:.4f}]  Precision: [{:.4f}, {:.4f}]  "
            "Recall: [{:.4f}, {:.4f}]  F1-score: [{:.4f}, {:.4f}]  AUC: [{:.4f}, {:.4f}]".format(
                np.mean(accs),
                np.std(accs),
                np.mean(precisions),
                np.std(precisions),
                np.mean(recalls),
                np.std(recalls),
                np.mean(f1s),
                np.std(f1s),
                np.mean(auc_score_list),
                np.std(auc_score_list),
            )
        )
        return (
            np.mean(accs),
            np.mean(precisions),
            np.mean(recalls),
            np.mean(f1s),
            np.mean(auc_score_list),
        )
    else:
        return np.mean(accs), np.mean(f1s)


def find_optimal_threshold(probs, labels):
    """Find optimal classification threshold"""
    best_acc = 0
    best_threshold = 0.5

    for threshold in np.arange(0.3, 0.7, 0.01):
        predictions = (probs[:, 1] >= threshold).float()
        accuracy = (predictions == labels).float().mean().item()

        if accuracy > best_acc:
            best_acc = accuracy
            best_threshold = threshold

    return best_threshold, best_acc


def process_per_value(per_value, model_type="mlp"):
    """Process data loading and evaluation for specific training ratio"""
    print(f"\n{'='*80}")
    print(f"Processing dataset {DATASET_NAME}_{per_value}0 (Model type: {model_type})")
    print(f"{'='*80}")

    url_embedding_list = []
    classifier_train_indices = []
    classifier_test_indices = []
    classifier_val_indices = []
    label_list = []
    node_counts = []  # Record node count for each subgraph

    # Load data for each subgraph
    for i in range(NUM_PARTS):
        print(f"Loading data for subgraph {i+1}/{NUM_PARTS}...")

        # Load embedding vectors
        embedding_path = f"./embeddings/phishscope_part{i}.pt"
        embedding = torch.load(embedding_path)
        url_embedding_list.append(embedding)
        node_counts.append(embedding.shape[0])  # Record node count for this subgraph

        # Load labels
        label_path = f"./data/phishscope/phishscope_part{i}/labels.npy"
        label = np.load(label_path).astype("int32")
        label = encode_onehot(label)
        label = torch.FloatTensor(label)
        label_list.append(label)

        # Load indices for specific training ratio
        train_indices_path = (
            f"./data/phishscope/phishscope_part{i}/train_{per_value}0.npy"
        )
        test_indices_path = (
            f"./data/phishscope/phishscope_part{i}/test_{per_value}0.npy"
        )
        val_indices_path = f"./data/phishscope/phishscope_part{i}/val_{per_value}0.npy"

        # Ensure files exist
        if not all(
            os.path.exists(p)
            for p in [train_indices_path, test_indices_path, val_indices_path]
        ):
            print(
                f"Warning: Data files for {DATASET_NAME}_{per_value}0 not found, skipping..."
            )
            return None

        classifier_train_indice = np.load(train_indices_path)
        classifier_test_indice = np.load(test_indices_path)
        classifier_val_indice = np.load(val_indices_path)

        classifier_train_indice = torch.LongTensor(classifier_train_indice)
        classifier_test_indice = torch.LongTensor(classifier_test_indice)
        classifier_val_indice = torch.LongTensor(classifier_val_indice)

        classifier_train_indices.append(classifier_train_indice)
        classifier_test_indices.append(classifier_test_indice)
        classifier_val_indices.append(classifier_val_indice)

    # Adjust local indices to global indices
    print("Converting local indices to global indices...")
    adjusted_train_indices = []
    adjusted_test_indices = []
    adjusted_val_indices = []
    offset = 0

    for i in range(NUM_PARTS):
        # Add cumulative offset to current subgraph indices
        adjusted_train_indices.append(classifier_train_indices[i] + offset)
        adjusted_test_indices.append(classifier_test_indices[i] + offset)
        adjusted_val_indices.append(classifier_val_indices[i] + offset)

        # Update offset to total number of processed nodes
        offset += node_counts[i]

    # Merge subgraph data
    print("Merging subgraph data...")
    combined_embeddings = torch.cat(url_embedding_list, dim=0)
    combined_labels = torch.cat(label_list, dim=0)
    combined_train_indices = torch.cat(adjusted_train_indices, dim=0)
    combined_test_indices = torch.cat(adjusted_test_indices, dim=0)
    combined_val_indices = torch.cat(adjusted_val_indices, dim=0)

    print(f"Combined embedding shape: {combined_embeddings.shape}")
    print(f"Combined label shape: {combined_labels.shape}")
    print(f"Number of combined training indices: {len(combined_train_indices)}")
    print(f"Number of combined test indices: {len(combined_test_indices)}")
    print(f"Number of combined validation indices: {len(combined_val_indices)}")

    # Move data to device
    combined_embeddings = combined_embeddings.to(device)
    combined_labels = combined_labels.to(device)
    combined_train_indices = combined_train_indices.to(device)
    combined_test_indices = combined_test_indices.to(device)
    combined_val_indices = combined_val_indices.to(device)

    # Determine number of classes
    nb_classes = combined_labels.shape[1]
    print(f"Number of classes: {nb_classes}")

    # Perform evaluation
    print(f"Starting overall evaluation on {DATASET_NAME}_{per_value}0...")
    accuracy, precision, recall, f1, auc = evaluate(
        combined_embeddings,
        combined_train_indices,
        combined_val_indices,
        combined_test_indices,
        combined_labels,
        nb_classes,
        device,
        model_type=model_type,  # Use specified model type
        lr=0.005 if model_type == "mlp" else 0.01,  # Lower learning rate for MLP
        wd=0.0001 if model_type == "mlp" else 0.0,  # Weight decay for MLP
        hidden_dims=[256, 128],  # Hidden layer dimensions for MLP
        dropout=0.3,  # Dropout rate
        patience=15,  # Early stopping patience
        per_value=per_value,  # Pass training ratio parameter
    )

    # Display results
    result = {
        "per": f"{DATASET_NAME}_{per_value}0",
        "model": model_type,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }

    print(f"\nEvaluation results for {DATASET_NAME}_{per_value}0 using {model_type}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy-AUC Gap: {auc-accuracy:.4f}")

    return result


# Main program
def main():
    print(f"Starting evaluation of merged subgraphs with different training ratios...")

    # Store all results
    all_results = {}

    # Model type selection
    model_types = ["mlp"]  # Can compare two model types

    for model_type in model_types:
        all_results[model_type] = {}

        # Process different training ratios in order
        for per_value in PER_VALUES:
            result = process_per_value(per_value, model_type=model_type)
            if result:
                all_results[model_type][per_value] = result

    # Summarize all results
    print("\n\nFinal results summary:")
    print(f"{'-'*130}")
    print(
        f"{'Model Type':<10} {'Train Ratio':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-score':<10} {'AUC':<10} {'ACC-AUC Gap':<10}"
    )
    print(f"{'-'*130}")

    for model_type in model_types:
        for per_value in PER_VALUES:
            if per_value in all_results[model_type]:
                result = all_results[model_type][per_value]
                gap = result["auc"] - result["accuracy"]
                print(
                    f"{result['model']:<10} {result['per']:<10} {result['accuracy']:<10.4f} "
                    f"{result['precision']:<10.4f} {result['recall']:<10.4f} {result['f1']:<10.4f} "
                    f"{result['auc']:<10.4f} {gap:<10.4f}"
                )

        # Add separator after each model
        print(f"{'-'*130}")

    print(
        f"\nAll URL classification information has been saved to files in the ./results/ directory"
    )


if __name__ == "__main__":
    main()
