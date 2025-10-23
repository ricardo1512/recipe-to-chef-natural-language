import csv
import ast

# Input and output files
input_file = 'Data/test-no-labels.csv'
output_file = 'Data/test-no-labels_preprocessed.csv'

def clean_list_string(text, column_name):
    """Clean list-like strings with special handling for steps, tags, and ingredients"""
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

# Load all rows from the input CSV
rows = []
with open(input_file, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=';')  # Usa nomes das colunas
    for row in reader:
        recipe_parts = []
        for col_name, value in row.items():
            if col_name.lower() in ['tags', 'steps', 'ingredients']:
                cleaned = clean_list_string(value, col_name)
            else:
                cleaned = value.strip()
            recipe_parts.append(cleaned)

        # Join all recipe components into a single string
        recipe_str = '; '.join(recipe_parts)
        rows.append([recipe_str])

# Export the preprocessed test set
with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile, delimiter=';')
    writer.writerow(['recipe'])
    writer.writerows(rows)

print(f"Test set successfully exported to: {output_file}")
print(f"Total samples: {len(rows)}")
