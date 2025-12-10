import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def find_metric_files(base_dir):
    """Finds all 'metrics_summary.json' files in the evaluation directory."""
    metric_files = []
    if not os.path.isdir(base_dir):
        print(f"Warning: Base directory '{base_dir}' not found.")
        return metric_files

    for model_name in os.listdir(base_dir):
        model_dir = os.path.join(base_dir, model_name)
        if os.path.isdir(model_dir):
            metric_file = os.path.join(model_dir, 'metrics_summary.json')
            if os.path.isfile(metric_file):
                metric_files.append((model_name, metric_file))
    return metric_files

def load_all_metrics(metric_files):
    """Loads all metrics from the found files into a single DataFrame."""
    all_metrics = []
    for model_name, file_path in metric_files:
        with open(file_path, 'r') as f:
            metrics = json.load(f)
            metrics['model_name'] = model_name
            all_metrics.append(metrics)
    
    if not all_metrics:
        return pd.DataFrame()
        
    return pd.DataFrame(all_metrics)

def plot_comparison(df, output_dir):
    """Plots a comparison of all metrics across all models."""
    if df.empty:
        print("No metrics data to plot.")
        return

    # Melt the DataFrame to make it suitable for seaborn's catplot
    df_melted = df.melt(id_vars='model_name', var_name='metric', value_name='score')
    
    # Get the number of metrics to adjust plot height
    num_metrics = len(df_melted['metric'].unique())
    
    # Create a categorical plot
    g = sns.catplot(
        data=df_melted,
        x='score',
        y='metric',
        hue='model_name',
        kind='bar',
        height=max(6, num_metrics * 0.5),  # Adjust height based on number of metrics
        aspect=1.5,
        legend_out=True,
        palette='viridis'
    )
    
    g.fig.suptitle('Model Performance Comparison', fontsize=16, y=1.02)
    g.set_axis_labels('Score', 'Metric')
    g.set_titles("Metric: {col_name}")
    
    # Add value labels to the bars
    for ax in g.axes.flat:
        for p in ax.patches:
            ax.text(p.get_width(), p.get_y() + p.get_height() / 2.,
                    f'{p.get_width():.3f}', 
                    ha='left', va='center', fontsize=9)

    # Adjust x-axis limits to give space for labels
    for ax in g.axes.flat:
        ax.set_xlim(0, ax.get_xlim()[1] * 1.15)

    plot_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to {plot_path}")


def main(args):
    """Main function to run the comparison."""
    metric_files = find_metric_files(args.eval_dir)
    if not metric_files:
        print("No model evaluation results found to compare.")
        return

    print("Found the following model metrics to compare:")
    for model_name, _ in metric_files:
        print(f"- {model_name}")

    metrics_df = load_all_metrics(metric_files)
    
    # Ensure the output directory for the comparison plot exists
    os.makedirs(args.eval_dir, exist_ok=True)
    
    # Save the combined metrics to a CSV file for easy inspection
    summary_csv_path = os.path.join(args.eval_dir, 'all_models_summary.csv')
    metrics_df.to_csv(summary_csv_path, index=False)
    print(f"\nCombined metrics saved to {summary_csv_path}")

    # Plot the comparison
    plot_comparison(metrics_df, args.eval_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare performance metrics across multiple evaluated LLMs.")
    parser.add_argument(
        '--eval_dir',
        type=str,
        default='../result/evaluation',
        help="The directory containing the model evaluation subdirectories."
    )
    
    args = parser.parse_args()
    main(args)
