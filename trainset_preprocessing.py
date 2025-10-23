import csv
import ast
import random
from collections import Counter

csv_file = 'train.csv'

# Output files
train_file = 'trainset_preprocessed.csv'
valid_file = 'validset_preprocessed.csv'
test_file = 'testset_preprocessed.csv'

# Split ratios
train_ratio = 0.7
valid_ratio = 0.15
test_ratio = 0.15


def clean_list_string(text, column_name):
    """Clean list-like strings with specialized handling for each column"""
    parsed = ast.literal_eval(text)
    col = column_name.lower()
    if col == 'steps':
        return '; '.join([f"{i + 1}. {str(step).strip().strip('\"\'')}" for i, step in enumerate(parsed)])
    elif col == 'tags':
        return ', '.join([str(tag).strip().strip('\"\'').replace('-', ' ') for tag in parsed])
    elif col == 'ingredients':
        return ', '.join([str(ingredient).strip().strip('\"\'') for ingredient in parsed])
    else:
        return text.strip()


def convert_chef_id(chef_id_str):
    """Convert chef_id to integer safely"""
    try:
        return int(chef_id_str.strip())
    except (ValueError, TypeError):
        return chef_id_str


# Counter to track samples per chef_id
chef_id_counter = Counter()

# Load all rows first
rows = []
with open(csv_file, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=';')  # Usando nomes das colunas
    for row in reader:
        chef_id = convert_chef_id(row['chef_id'])
        chef_id_counter[chef_id] += 1

        recipe_parts = []
        for col_name, value in row.items():
            if col_name.lower() in ['steps', 'tags', 'ingredients']:
                cleaned = clean_list_string(value, col_name)
            elif col_name.lower() != 'chef_id':
                cleaned = value.strip()
            else:
                continue
            recipe_parts.append(cleaned)

        recipe_str = '; '.join(recipe_parts)
        rows.append([chef_id, recipe_str])

# Split into train/valid/test
total = len(rows)
train_end = int(total * train_ratio)
valid_end = train_end + int(total * valid_ratio)

train_rows = rows[:train_end]
valid_rows = rows[train_end:valid_end]
test_rows = rows[valid_end:]


def export_csv(filename, data):
    """Helper function to export CSV file"""
    with open(filename, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile, delimiter=';')
        writer.writerow(['chef_id', 'recipe'])
        writer.writerows(data)


# Export datasets
export_csv(train_file, train_rows)
export_csv(valid_file, valid_rows)
export_csv(test_file, test_rows)

print(f"Files successfully exported to:\n - {train_file}\n - {valid_file}\n - {test_file}\n")

print("=" * 50)
print("SAMPLES PER CHEF_ID (FULL DATASET):")
print("=" * 50)
for chef_id, count in chef_id_counter.most_common():
    print(f"Chef ID {chef_id}: {count} samples")

print("\n" + "=" * 50)
print("SUMMARY STATISTICS:")
print("=" * 50)
print(f"Total samples: {sum(chef_id_counter.values())}")
print(f"Number of unique chef_ids: {len(chef_id_counter)}")
print(f"Most common chef_id: {chef_id_counter.most_common(1)[0][0]} "
      f"with {chef_id_counter.most_common(1)[0][1]} samples")
print(f"Least common chef_id: {chef_id_counter.most_common()[-1][0]} "
      f"with {chef_id_counter.most_common()[-1][1]} sample(s)")
