import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
import matplotlib.pyplot as plt

print(torch.cuda.is_available())

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
        label = self.labels[idx]
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
# Main
# ======================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load dataset
    print("Loading data...")
    train_set = pd.read_csv('Data/trainset_preprocessed.csv', sep=';', header=0, names=['chef_id', 'recipe'])
    valid_set = pd.read_csv('Data/validset_preprocessed.csv', sep=';', header=0, names=['chef_id', 'recipe'])
    test_set = pd.read_csv('Data/testset_preprocessed.csv', sep=';', header=0, names=['chef_id', 'recipe'])
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
    print(f"Classes: {label_encoder.classes_}")

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
    train_dataset = RecipeDataset(train_set['recipe'].tolist(), train_labels, tokenizer)
    val_dataset = RecipeDataset(valid_set['recipe'].tolist(), val_labels, tokenizer)
    test_dataset = RecipeDataset(test_set['recipe'].tolist(), test_labels, tokenizer)
    print("Datasets created.")

    # Set training arguments
    print("Setting up training arguments...")
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
        logging_dir='./logs',
        logging_steps=10,
    )
    print("Training arguments set.")

    # Initialize Trainer
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    print("Trainer initialized.")

    # Start training
    print("Starting training...")
    trainer.train()
    print("Training finished.")

    # Plot loss curve
    print("Plotting loss curve...")
    plot_loss_curve(trainer.state.log_history, "Images/loss_curve_bert.png")

    # Evaluate on validation set
    print("Evaluating on validation set...")
    val_results = trainer.evaluate()
    val_acc = val_results['eval_accuracy']
    print(f"Validation Accuracy: {val_acc:.4f}")

    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.predict(test_dataset)
    test_acc = accuracy_score(test_labels, np.argmax(test_results.predictions, axis=-1))
    print(f"Test Accuracy: {test_acc:.4f}")

    # Export results to CSV
    results_df = pd.DataFrame([{
        "model": "BERT",
        "valid_accuracy": val_acc,
        "test_accuracy": test_acc
    }])
    results_csv_path = "Data/results_summary.csv"

    if not os.path.exists(results_csv_path):
        results_df.to_csv(results_csv_path, mode='w', header=True, index=False)
    else:
        results_df.to_csv(results_csv_path, mode='a', header=False, index=False)

    print(f"Results saved/appended to {results_csv_path}")


if __name__ == "__main__":
    main()
