import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ======================
# Dataset class
# ======================
class RecipeDataset(Dataset):
    def __init__(self, texts, labels, vocab=None, max_length=200):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        if vocab is None:
            self.build_vocab()
        else:
            self.vocab = vocab
        self.vocab_size = len(self.vocab) + 1  # +1 for padding index 0

    def build_vocab(self):
        all_tokens = set()
        for text in self.texts:
            tokens = text.lower().split()
            all_tokens.update(tokens)
        self.vocab = {tok: i+1 for i, tok in enumerate(sorted(all_tokens))}  # 0 = padding

    def text_to_seq(self, text):
        tokens = text.lower().split()
        seq = [self.vocab.get(tok, 0) for tok in tokens]
        if len(seq) < self.max_length:
            seq += [0] * (self.max_length - len(seq))
        else:
            seq = seq[:self.max_length]
        return torch.tensor(seq, dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        seq = self.text_to_seq(self.texts[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return seq, label


# ======================
# Bidirectional LSTM Model
# ======================
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 num_layers=2, dropout=0.3, pretrained_embeddings=None):
        super(LSTMClassifier, self).__init__()
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(pretrained_embeddings), freeze=False, padding_idx=0
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        output, (hn, cn) = self.lstm(x)
        forward_hn = hn[-2, :, :]
        backward_hn = hn[-1, :, :]
        hn_combined = torch.cat((forward_hn, backward_hn), dim=1)
        out = self.fc(hn_combined)
        return out


# ======================
# Training function (save best model)
# ======================
def train_model(model, train_loader, val_loader, num_epochs=15, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

    train_losses, val_losses, val_accuracies = [], [], []

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(1, num_epochs+1):
        model.train()
        running_loss = 0
        for seqs, labels in train_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * seqs.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_running_loss = 0
        val_preds, val_labels_list = [], []
        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs, labels = seqs.to(device), labels.to(device)
                outputs = model(seqs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * seqs.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())
        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = accuracy_score(val_labels_list, val_preds)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()

        scheduler.step(val_loss)
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

    # Load best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return train_losses, val_losses, val_accuracies, model, best_val_acc


# ======================
# Loss plotting
# ======================
def plot_loss_curve(train_losses, val_losses, filename):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10,6))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    epochs = list(range(1, len(train_losses)+1))
    ax.plot(epochs, train_losses, label='Training Loss', color='cyan', marker='o')
    ax.plot(epochs, val_losses, label='Validation Loss', color='magenta', marker='s')
    ax.set_xlabel('Epoch', color='white')
    ax.set_ylabel('Loss', color='white')
    # ax.set_title('Training and Validation Loss Over Epochs - BiLSTM', color='white')
    ax.tick_params(colors='white')
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5)
    ax.legend(loc='upper right', facecolor='black', edgecolor='white', labelcolor='white')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Loss curve saved to {filename}")


# ======================
# Load GloVe embeddings
# ======================
def load_glove_embeddings(glove_file, vocab, embed_dim=100):
    embeddings_index = {}
    with open(glove_file, encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    embedding_matrix = np.random.normal(scale=0.6, size=(len(vocab)+1, embed_dim))
    for word, i in vocab.items():
        vector = embeddings_index.get(word)
        if vector is not None:
            embedding_matrix[i] = vector
    return embedding_matrix


# ======================
# Main
# ======================
def main():
    # Load dataset
    print("Loading data...")
    train_set = pd.read_csv('trainset_preprocessed.csv', sep=';', header=0, names=['chef_id', 'recipe'])
    valid_set = pd.read_csv('validset_preprocessed.csv', sep=';', header=0, names=['chef_id', 'recipe'])
    test_set = pd.read_csv('testset_preprocessed.csv', sep=';', header=0, names=['chef_id', 'recipe'])
    print(f"Training samples: {len(train_set)}")
    print(f"Validation samples: {len(valid_set)}")
    print(f"Test samples: {len(test_set)}")

    # Encode labels
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_set['chef_id'])
    val_labels = label_encoder.transform(valid_set['chef_id'])
    test_labels = label_encoder.transform(test_set['chef_id'])
    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")

    # Create datasets
    train_dataset = RecipeDataset(train_set['recipe'].tolist(), train_labels)
    val_dataset = RecipeDataset(valid_set['recipe'].tolist(), val_labels, vocab=train_dataset.vocab)
    test_dataset = RecipeDataset(test_set['recipe'].tolist(), test_labels, vocab=train_dataset.vocab)

    # Load GloVe embeddings
    glove_file = "glove.6B.100d.txt"
    pretrained_embeddings = load_glove_embeddings(glove_file, train_dataset.vocab, embed_dim=100)

    # Create model
    model = LSTMClassifier(
        vocab_size=train_dataset.vocab_size,
        embed_dim=100,
        hidden_dim=256,
        num_classes=num_classes,
        num_layers=2,
        dropout=0.3,
        pretrained_embeddings=pretrained_embeddings
    ).to(device)

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Train
    train_losses, val_losses, val_accuracies, model, best_val_acc = train_model(
        model, train_loader, val_loader, num_epochs=15, lr=1e-3
    )

    # Plot loss
    plot_loss_curve(train_losses, val_losses, "loss_curve_bilstm.png")

    # Evaluate test
    model.eval()
    test_preds, test_labels_list = [], []
    with torch.no_grad():
        for seqs, labels in test_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            outputs = model(seqs)
            preds = torch.argmax(outputs, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels_list.extend(labels.cpu().numpy())
    test_acc = accuracy_score(test_labels_list, test_preds)
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Export results to CSV
    results_df = pd.DataFrame([{
        "model": "BiLSTM",
        "valid_accuracy": best_val_acc,
        "test_accuracy": test_acc
    }])
    results_csv_path = "results_summary.csv"
    if not os.path.exists(results_csv_path):
        results_df.to_csv(results_csv_path, mode='w', header=True, index=False)
    else:
        results_df.to_csv(results_csv_path, mode='a', header=False, index=False)

    print(f"Results saved/appended to {results_csv_path}")


if __name__ == "__main__":
    main()
