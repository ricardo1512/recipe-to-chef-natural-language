import os
import pandas as pd
import numpy as np
import re
import random
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import wordnet

print(torch.cuda.is_available())

# Download WordNet once
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


# ======================
# Data augmentation
# ======================
def augment_recipe(text, replace_prob=0.2):
    """
    Augment recipe text by replacing words with synonyms while preserving numbers.
    """
    numbers = re.findall(r'\d+[\.,]?\d*', text)
    temp_text = re.sub(r'\d+[\.,]?\d*', lambda m: f"<NUM_{numbers.index(m.group())}>", text)

    words = temp_text.split()
    new_words = []

    for w in words:
        if not w.startswith("<NUM_") and random.random() < replace_prob:
            syns = wordnet.synsets(w.lower())
            if syns:
                lemmas = [l.name().replace('_', ' ') for s in syns for l in s.lemmas()]
                lemmas = list(dict.fromkeys(lemmas))
                lemmas = [lm for lm in lemmas if lm.lower() != w.lower()]
                if lemmas:
                    choice = random.choice(lemmas)
                    if w[0].isupper():
                        choice = choice.capitalize()
                    new_words.append(choice)
                    continue
        new_words.append(w)

    aug_text = ' '.join(new_words)
    for i, num in enumerate(numbers):
        aug_text = aug_text.replace(f"<NUM_{i}>", num)

    return aug_text


def balance_augment_texts(texts, labels, label_encoder, augment_fn, replace_prob=0.2, seed=42):
    """
    Balance classes by augmenting texts until each class reaches the size of the largest class.
    """
    random.seed(seed)
    np.random.seed(seed)

    texts = list(texts)
    labels = list(labels)

    counts = Counter(labels)
    max_count = max(counts.values())

    print("Original training class distribution:")
    for cls, cnt in sorted(counts.items()):
        class_name = label_encoder.inverse_transform([cls])[0]
        print(f"  Class {cls} ({class_name}): {cnt} samples")
    print(f"Majority class size (target): {max_count}")

    augmented_texts = []
    augmented_labels = []

    indices_by_class = {}
    for idx, lbl in enumerate(labels):
        indices_by_class.setdefault(lbl, []).append(idx)

    for cls, idx_list in indices_by_class.items():
        current_count = len(idx_list)
        deficit = max_count - current_count
        if deficit <= 0:
            continue
        sampled_indices = np.random.choice(idx_list, size=deficit, replace=True)
        for si in sampled_indices:
            original_text = texts[si]
            aug_text = augment_fn(original_text, replace_prob=replace_prob)
            augmented_texts.append(aug_text)
            augmented_labels.append(cls)

    new_texts = texts + augmented_texts
    new_labels = labels + augmented_labels

    new_counts = Counter(new_labels)
    print("Class distribution after balancing augmentation:")
    for cls, cnt in sorted(new_counts.items()):
        class_name = label_encoder.inverse_transform([cls])[0]
        print(f"  Class {cls} ({class_name}): {cnt} samples")

    return new_texts, new_labels


# ======================
# Dataset class
# ======================
class RecipeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label)
        return item


# ======================
# Metrics
# ======================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    print(f"Computed Accuracy: {acc:.4f}")
    return {'accuracy': acc}


# ======================
# Confusion Matrix
# ======================
def plot_confusion_matrix(y_true, y_pred, label_encoder, title="Confusion Matrix", figsize=(12, 10), save_path=None):
    """
    Plot a confusion matrix showing both absolute numbers and percentages.
    """
    # Compute the raw confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    labels = label_encoder.classes_

    # Compute percentages for each cell (row-wise)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Prepare annotation text combining counts and percentages
    annot = np.empty_like(cm).astype(str)
    n_rows, n_cols = cm.shape
    for i in range(n_rows):
        for j in range(n_cols):
            annot[i, j] = f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)"

    # Set dark background style
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap using percentages for color intensity
    # Absolute numbers are shown in the annotations
    sns.heatmap(cm_percent, annot=annot, fmt='', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)

    # Axis labels and title
    ax.set_xlabel('Predicted', color='white')
    ax.set_ylabel('True', color='white')
    ax.set_title(title, color='white')

    # Rotate tick labels for better readability
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Confusion matrix saved to {save_path}")



# ======================
# Loss curve plotting
# ======================
def plot_loss_curve(log_history, filename):
    train_losses = [entry['loss'] for entry in log_history if 'loss' in entry and 'epoch' in entry]
    eval_losses = [entry['eval_loss'] for entry in log_history if 'eval_loss' in entry and 'epoch' in entry]
    epochs_train = [entry['epoch'] for entry in log_history if 'loss' in entry and 'epoch' in entry]
    epochs_eval = [entry['epoch'] for entry in log_history if 'eval_loss' in entry and 'epoch' in entry]

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    ax.plot(epochs_train, train_losses, label='Training Loss', color='cyan', marker='o')
    ax.plot(epochs_eval, eval_losses, label='Validation Loss', color='magenta', marker='s')

    ax.set_xlabel('Epoch', color='white')
    ax.set_ylabel('Loss', color='white')
    # ax.set_title('Training and Validation Loss Over Epochs - BERT', color='white')

    ax.tick_params(colors='white')
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5)
    ax.legend(loc='upper right', facecolor='black', edgecolor='white', labelcolor='white')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Loss curve saved to {filename}")


# ======================
# Prediction
# ======================
def predict_test_no_labels(model, tokenizer, label_encoder, input_file='test-no-labels_preprocessed.csv', output_file='results.txt', device='cuda'):
    """
    Predict labels for a CSV file with only 'recipe' column and export predictions to a text file.
    """
    # Load test data
    test_df = pd.read_csv(input_file, sep=';', header=0, names=['recipe'])
    test_texts = test_df['recipe'].tolist()

    # Create dataset
    test_dataset = RecipeDataset(test_texts, [0]*len(test_texts), tokenizer)  # labels dummy

    # Predict
    model.eval()
    predictions = []
    for item in test_dataset:
        inputs = {k: v.unsqueeze(0).to(device) for k, v in item.items() if k != 'labels'}
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=-1).item()
            predictions.append(pred)

    # Decode labels
    decoded_preds = label_encoder.inverse_transform(predictions)

    # Save to txt
    with open(output_file, 'w') as f:
        for label in decoded_preds:
            f.write(f"{label}\n")

    print(f"Inference completed. Predictions saved to {output_file}")


# ======================
# Main
# ======================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load pre-split datasets
    print("Loading data...")
    train_set = pd.read_csv('trainset_preprocessed.csv', sep=';', header=0, names=['chef_id', 'recipe'])
    valid_set = pd.read_csv('validset_preprocessed.csv', sep=';', header=0, names=['chef_id', 'recipe'])
    test_set = pd.read_csv('testset_preprocessed.csv', sep=';', header=0, names=['chef_id', 'recipe'])
    print(f"Training samples: {len(train_set)}")
    print(f"Validation samples: {len(valid_set)}")
    print(f"Test samples: {len(test_set)}")

    # Encode labels
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_set['chef_id'])
    val_labels = label_encoder.transform(valid_set['chef_id'])
    test_labels = label_encoder.transform(test_set['chef_id'])
    print(f"Number of classes: {len(label_encoder.classes_)}")

    # Apply augmentation ONLY to the TRAINING set
    print("Applying balancing augmentation ONLY to the TRAINING set...")
    train_texts_aug, train_labels_aug = balance_augment_texts(
        train_set['recipe'].tolist(), train_labels, label_encoder, augment_recipe, replace_prob=0.2, seed=42
    )

    # Load tokenizer and model
    print("Loading tokenizer and model...")
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_encoder.classes_)
    ).to(device)
    print("Tokenizer and model loaded.")

    # Create datasets
    print("Creating datasets...")
    train_dataset = RecipeDataset(train_texts_aug, train_labels_aug, tokenizer)
    val_dataset = RecipeDataset(valid_set['recipe'].tolist(), val_labels, tokenizer)
    test_dataset = RecipeDataset(test_set['recipe'].tolist(), test_labels, tokenizer)
    print("Datasets created.")

    # Set training arguments
    print("Setting training arguments...")
    training_args = TrainingArguments(
        output_dir='./results',
        eval_strategy='epoch',
        save_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        logging_steps=10,
    )

    # Initialize Trainer
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # Start training
    print("Starting training...")
    trainer.train()
    print("Training completed.")

    # Plot loss curve
    print("Plotting loss curve...")
    plot_loss_curve(trainer.state.log_history, "loss_curve_bert_augmentation.png")

    # Validation
    print("Evaluating on validation set...")
    val_results = trainer.evaluate()
    val_acc = val_results['eval_accuracy']
    print(f"Validation Accuracy: {val_acc:.4f}")

    # Test
    print("Evaluating on test set...")
    test_results = trainer.predict(test_dataset)
    test_preds = np.argmax(test_results.predictions, axis=-1)
    test_acc = accuracy_score(test_labels, test_preds)
    print(f"Test Accuracy: {test_acc:.4f}")


    # ======================
    # Export prediction errors (failures)
    # ======================
    print("Exporting prediction failures...")

    # Decode labels and predictions
    true_labels_decoded = label_encoder.inverse_transform(test_labels)
    pred_labels_decoded = label_encoder.inverse_transform(test_preds)

    # Combine test data into a single DataFrame
    failures_df = pd.DataFrame({
        'true_label': true_labels_decoded,
        'predicted_label': pred_labels_decoded,
        'recipe': test_set['recipe'].tolist()
    })

    # Filter only the incorrect predictions
    failures_df = failures_df[failures_df['true_label'] != failures_df['predicted_label']]

    # Save one CSV per true class (actual label)
    for cls in sorted(failures_df['true_label'].unique()):
        cls_failures = failures_df[failures_df['true_label'] == cls]
        file_path = os.path.join("failed_predictions", f"failures_{cls}.csv")
        cls_failures.to_csv(file_path, sep=';', index=False)
        print(f"  → Saved {len(cls_failures)} failures for true class '{cls}' to {file_path}")


    # ======================
    # Plot confusion matrix for test set
    # ======================
    print("Plotting confusion matrix for test set...")
    plot_confusion_matrix(test_labels, test_preds, label_encoder, title="Test Set Confusion Matrix", save_path="confusion_matrix_best_model.png")

    # ======================
    # Export results to CSV
    # ======================
    results_df = pd.DataFrame([{
        "model": "BERT Augmentation",
        "valid_accuracy": val_acc,
        "test_accuracy": test_acc,
    }])
    results_csv_path = "results_summary.csv"
    results_df.to_csv(
        results_csv_path,
        mode='a',
        header=not os.path.exists(results_csv_path),
        index=False
    )


    # ======================
    # Predict test-no-labels CSV
    # ======================
    predict_test_no_labels(model, tokenizer, label_encoder,
                           input_file='test-no-labels_preprocessed.csv',
                           output_file='results.txt',
                           device=device)


if __name__ == "__main__":
    main()
