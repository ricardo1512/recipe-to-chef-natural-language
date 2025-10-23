# Recipe-to-Chef Prediction Using Natural Language Processing

## Overview

This Natural Language Processing project aims to automatically associate recipes with their corresponding chefs. The task is challenging due to noisy data, unbalanced class distributions, and inconsistencies in textual formatting. We explore classical machine learning methods and deep learning strategies to classify recipes and identify stylistic patterns unique to each chef.

---

## Author
**Ricardo de Jesus Vicente Tavares**  
Instituto Superior Técnico, Universidade de Lisboa,  
October 2025

---

## Dataset

* The original dataset is provided in `train.csv`.
* Preprocessing steps:

  * The raw dataset contained multiple fields:
    - `chef_id` (label), 
    - `recipe_name`, 
    - `data`, `tags`, 
    - `steps`, 
    - `description`, 
    - `ingredients`, 
    - `n_ingredients`.
  * A new column **`recipe`** was created by concatenating the cleaned text from these fields (except the labels), separated by semicolons.
  * List-like fields (`tags`, `steps`, and `ingredients`) were flattened into plain text with consistent formatting.
  * `step` lists were explicitly numbered during preprocessing (1., 2., 3., …).
  * The final dataset contains exactly two columns:  
    - `chef_id` → the label.  
    - `recipe` → the preprocessed text used as model input.
* Data splits:

  * 70% Training
  * 15% Validation
  * 15% Test
* All subsets are exported as separate CSV files.

---

## Models

Three classification approaches were implemented:

1. **Support Vector Classifier (SVC)**

   * Recipes converted to TF-IDF vectors.
   * Linear kernel, probability estimates enabled.

2. **BiLSTM Classifier**

   * Initialized with pretrained GloVe embeddings (100 dimensions).
   * Bidirectional LSTM with two layers, dropout of 0.3.
   * Sequence length 200 tokens, batch size 32, 15 epochs.

3. **BERT-based Classifier**

   * Uses `bert-base-uncased` pretrained model.
   * Tokenized with `BertTokenizerFast`.
   * Maximum sequence length 512 tokens, batch size 8, 5 epochs.
   * Optimizer: AdamW with learning rate `2e-5` and weight decay `0.01`.

### Data Augmentation

* Synonym replacement using WordNet applied only to the training set.
* Numeric quantities masked to prevent altering ingredient measurements.
* Helps mitigate class imbalance and improve generalization.

---

## Experimental Setup

* **Metrics:** Accuracy on validation and test sets.
* **Hardware:** Experiments run on GPU.
* **Hyperparameters:**

  * SVC: Max 5000 TF-IDF features, linear kernel.
  * BiLSTM: Hidden size 256, 2 layers, dropout 0.3, 15 epochs, batch 32.
  * BERT: Max sequence 512, 5 epochs, learning rate 2e-5, batch size 8.

---

## Requirements

### Libraries
* **pandas**, **numpy** → data manipulation
* **scikit-learn** → LabelEncoder, metrics, TF‑IDF, SVC
* **torch**, **torchvision**, **torchaudio** → PyTorch and utilities
* **transformers** → BERT and tokenization
* **nltk** → WordNet and other NLP tools
* **matplotlib**, **seaborn** → visualization


```
pip install pandas numpy scikit-learn torch torchvision torchaudio transformers matplotlib seaborn nltk
```

### Files

  * **glove.6B.100d.txt** – This file is required for the project but is too large to include in the repository. You need to download it manually from the [GloVe website](https://nlp.stanford.edu/projects/glove/) and place it in the project folder.



---

## Run

### Data Visualization
```
python plot_classes_original_dataset.py
```
* **Output:**
  * `class_distribution_original.png`

### Preprocessing

#### Train Set
```
python trainset_preprocessing.py
```
* **Input:**
  * `train.csv`
* **Outputs:**
  * `trainset_preprocessed.csv`
  * `validset_preprocessed.csv`
  * `testset_preprocessed.csv`

#### Test Set
```
python test-no-labels_preprocessing.py
```
* **Input:**
  * `test-no-labels.csv`
* **Output:**
  * `test-no-labels_preprocessed.csv`

### Models

#### 1. SVC

##### Baseline
```
python svc.py
```
* **Inputs:**
  * `trainset_preprocessed.csv`
  * `validset_preprocessed.csv`
  * `testset_preprocessed.csv`
* **Output:**
  * `results_summary.csv`

##### With Data Augmentation
```
python svc_augmentation.py
```
* **Inputs:**
  * `trainset_preprocessed.csv`
  * `validset_preprocessed.csv`
  * `testset_preprocessed.csv`
* **Output:**
  * `results_summary.csv`

#### 2. BiLSTM

##### Baseline
```
python bilstm.py
```
* **Inputs:**
  * `trainset_preprocessed.csv`
  * `validset_preprocessed.csv`
  * `testset_preprocessed.csv`
* **Outputs:**
  * `loss_curve_bilstm.png`
  * `results_summary.csv`

##### With Data Augmentation
```
python bilstm_augmentation.py
```
* **Inputs:**
  * `trainset_preprocessed.csv`
  * `validset_preprocessed.csv`
  * `testset_preprocessed.csv`
* **Outputs:**
  * `loss_curve_bilstm_augmentation.png`
  * `results_summary.csv`

#### 3. BERT

##### Baseline
```
python bert.py
```
* **Inputs:**
  * `trainset_preprocessed.csv`
  * `validset_preprocessed.csv`
  * `testset_preprocessed.csv`
* **Outputs:**
  * `loss_curve_bert.png`
  * `results_summary.csv`

##### With Data Augmentation
```
python bert_augmentation.py
```
* **Inputs:**
  * `trainset_preprocessed.csv`
  * `validset_preprocessed.csv`
  * `testset_preprocessed.csv`
* **Outputs:**
  * `loss_curve_bert_augmentation.png`
  * `confusion_matrix_best_model.png`
  * `results_summary.csv`
  * `label_classes.npy`
  * `./results/bert_model/`
  * `./failed_predictions/failures_*.csv`
  * `results.txt`

### Results Visualization
```
python plot_results_models.py
```
* **Input:**
  * `results_summary.csv`
* **Output:**
  * `models_accuracies.png`

---
