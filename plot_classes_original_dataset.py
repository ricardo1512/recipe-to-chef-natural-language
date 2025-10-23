import pandas as pd
import matplotlib.pyplot as plt

def plot_genre_distribution(file_path, save_path='class_distribution_original.png'):
    # Load the dataset
    df = pd.read_csv(file_path, sep=';', header=0)

    # Count number of entries per genre
    genre_counts = df['chef_id'].astype(str).value_counts()

    # Print number of elements per class
    print("Number of recipes per chef:\n")
    print(genre_counts)

    # Create the figure with black background
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Choose a colormap for vibrant colors
    cmap = plt.colormaps['tab10']
    colors = [cmap(i % 10) for i in range(len(genre_counts))]

    # Plot the bar chart
    bars = ax.bar(genre_counts.index, genre_counts.values,
                  color=colors, edgecolor='white')

    # Set labels and title
    ax.set_xlabel('Chef ID', color='white')
    ax.set_ylabel('Number of Entries', color='white')
    # ax.set_title('Number of Recipes per Chef', color='white')

    # Rotate x-axis labels
    plt.xticks(rotation=0, ha='right', color='white')
    plt.yticks(color='white')

    # Add grid with gray lines
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor())
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    plot_genre_distribution('train.csv')
