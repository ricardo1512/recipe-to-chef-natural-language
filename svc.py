import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC


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
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_set['chef_id'])
    val_labels = label_encoder.transform(valid_set['chef_id'])
    test_labels = label_encoder.transform(test_set['chef_id'])
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Classes: {label_encoder.classes_}")

    # Convert texts to TF-IDF features
    print("Vectorizing text data with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_set['recipe'])
    X_val = vectorizer.transform(valid_set['recipe'])
    X_test = vectorizer.transform(test_set['recipe'])

    # Initialize SVC
    print("Initializing SVC...")
    model = SVC(kernel='linear', probability=True, random_state=42)

    # Train SVC
    print("Training SVC...")
    model.fit(X_train, train_labels)

    # Evaluate
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)

    train_acc = accuracy_score(train_labels, train_preds)
    val_acc = accuracy_score(val_labels, val_preds)
    test_acc = accuracy_score(test_labels, test_preds)

    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Export results
    results_df = pd.DataFrame([{
        "model": "SVC",
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
