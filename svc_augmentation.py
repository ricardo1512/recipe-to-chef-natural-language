import os
import pandas as pd
import numpy as np
import re
import random
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import nltk
from nltk.corpus import wordnet

# Download WordNet if not present
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


# ======================
# Data augmentation
# ======================
def augment_recipe(text, replace_prob=0.2):
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
    random.seed(seed)
    np.random.seed(seed)
    texts = list(texts)
    labels = list(labels)
    counts = Counter(labels)
    max_count = max(counts.values())

    augmented_texts = []
    augmented_labels = []

    indices_by_class = {}
    for idx, lbl in enumerate(labels):
        indices_by_class.setdefault(lbl, []).append(idx)

    for cls, idx_list in indices_by_class.items():
        deficit = max_count - len(idx_list)
        if deficit <= 0:
            continue
        sampled_indices = np.random.choice(idx_list, size=deficit, replace=True)
        for si in sampled_indices:
            aug_text = augment_fn(texts[si], replace_prob=replace_prob)
            augmented_texts.append(aug_text)
            augmented_labels.append(cls)

    new_texts = texts + augmented_texts
    new_labels = labels + augmented_labels
    return new_texts, new_labels


# ======================
# Main
# ======================
def main():
    print("Loading data...")
    train_set = pd.read_csv('Data/trainset_preprocessed.csv', sep=';', header=0, names=['chef_id', 'recipe'])
    valid_set = pd.read_csv('Data/validset_preprocessed.csv', sep=';', header=0, names=['chef_id', 'recipe'])
    test_set = pd.read_csv('Data/testset_preprocessed.csv', sep=';', header=0, names=['chef_id', 'recipe'])
    print(f"Training samples: {len(train_set)}")
    print(f"Validation samples: {len(valid_set)}")
    print(f"Test samples: {len(test_set)}")

    # Encode labels
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_set['chef_id'])
    val_labels = label_encoder.transform(valid_set['chef_id'])
    test_labels = label_encoder.transform(test_set['chef_id'])
    print(f"Number of classes: {len(label_encoder.classes_)}")

    # Apply augmentation to training set
    train_texts_aug, train_labels_aug = balance_augment_texts(
        train_set['recipe'].tolist(), train_labels, label_encoder, augment_recipe, replace_prob=0.2, seed=42
    )

    # Vectorize texts
    print("Vectorizing texts using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_texts_aug)
    X_val = vectorizer.transform(valid_set['recipe'])
    X_test = vectorizer.transform(test_set['recipe'])

    # Initialize SVC
    model = SVC(kernel='linear', probability=True, random_state=42)

    # Train
    print("Training SVC...")
    model.fit(X_train, train_labels_aug)

    # Evaluate
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)

    train_acc = accuracy_score(train_labels_aug, train_preds)
    val_acc = accuracy_score(val_labels, val_preds)
    test_acc = accuracy_score(test_labels, test_preds)

    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Export results
    results_df = pd.DataFrame([{
        "model": "SVC Augmentation",
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
