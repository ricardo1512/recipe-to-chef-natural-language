import pandas as pd
import matplotlib.pyplot as plt


def plot_model_accuracies(file_path, save_path='Images/models_accuracies.png'):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Check that required columns exist
    required_cols = ['model', 'valid_accuracy', 'test_accuracy']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV file must contain columns: {required_cols}")

    print("Model accuracies:\n", df)

    # Create the figure with black background
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Bar width and positions
    bar_width = 0.35
    indices = range(len(df))

    # Colors for groups
    valid_color = '#1f77b4'  # blue
    test_color = '#ff7f0e'  # orange

    # Plot bars side by side
    ax.bar([i - bar_width / 2 for i in indices], df['valid_accuracy'],
           width=bar_width, color=valid_color, edgecolor='white', label='Validation Accuracy')
    ax.bar([i + bar_width / 2 for i in indices], df['test_accuracy'],
           width=bar_width, color=test_color, edgecolor='white', label='Test Accuracy')

    # Set labels and title
    ax.set_xlabel('Model', color='white')
    ax.set_ylabel('Accuracy', color='white')
    # ax.set_title('Validation vs Test Accuracy per Model', color='white')

    # Center model names below the two bars
    ax.set_xticks(indices)
    ax.set_xticklabels(df['model'], rotation=0, ha='center', color='white')
    plt.yticks(color='white')

    # Add grid
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5, axis='y')

    # Add legend
    ax.legend(facecolor='black', edgecolor='white', labelcolor='white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor())
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    plot_model_accuracies('Data/results_summary.csv')
