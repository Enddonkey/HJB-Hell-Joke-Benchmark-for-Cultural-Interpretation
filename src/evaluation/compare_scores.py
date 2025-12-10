import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def find_summary_files(scoring_base_dir):
    """Finds all scoring_summary.json files in the scoring directory."""
    summary_files = []
    for model_dir in os.listdir(scoring_base_dir):
        full_path = os.path.join(scoring_base_dir, model_dir)
        if os.path.isdir(full_path):
            summary_file = os.path.join(full_path, 'scoring_summary.json')
            if os.path.exists(summary_file):
                summary_files.append((model_dir, summary_file))
    return summary_files

def aggregate_scores(summary_files):
    """Aggregates average scores from multiple summary files into a single DataFrame."""
    all_scores = []
    for model_name, file_path in summary_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            avg_scores = data.get('average_scores', {})
            avg_scores['model'] = model_name
            all_scores.append(avg_scores)
    
    if not all_scores:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_scores)
    # Reorder columns to have model first
    cols = ['model'] + [col for col in df.columns if col != 'model']
    return df[cols]

def plot_comparison_chart(df, output_dir):
    """Plots a grouped bar chart comparing model scores."""
    if df.empty:
        print("No data to plot for model comparison.")
        return

    df_melted = df.melt(id_vars='model', var_name='score_type', value_name='average_score')
    
    plt.figure(figsize=(14, 8))
    sns.barplot(data=df_melted, x='score_type', y='average_score', hue='model', palette='viridis')
    
    plt.title('Model Performance Comparison', fontsize=18)
    plt.xlabel('Score Dimension', fontsize=12)
    plt.ylabel('Average Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Model')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Model comparison chart saved to {plot_path}")

def plot_correlation_heatmap(df, output_dir):
    """Calculates and plots the correlation heatmap of score types."""
    if df.empty or df.shape[1] <= 2: # Need at least 2 score columns
        print("Not enough data to generate a correlation heatmap.")
        return
        
    score_cols = [col for col in df.columns if col != 'model']
    correlation_matrix = df[score_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    
    plt.title('Correlation Matrix of Scoring Dimensions', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'score_correlation_heatmap.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Correlation heatmap saved to {plot_path}")

def main(args):
    """Main function to run the comparison."""
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        summary_files = find_summary_files(args.scoring_base_dir)
        if not summary_files:
            print(f"No 'scoring_summary.json' files found in subdirectories of '{args.scoring_base_dir}'.")
            return
            
        print(f"Found {len(summary_files)} model summaries to compare.")
        
        aggregated_df = aggregate_scores(summary_files)
        
        # Save aggregated data
        aggregated_csv_path = os.path.join(args.output_dir, 'all_models_scores.csv')
        aggregated_df.to_csv(aggregated_csv_path, index=False, encoding='utf-8-sig')
        print(f"Aggregated scores saved to {aggregated_csv_path}")
        
        # Generate plots
        plot_comparison_chart(aggregated_df, args.output_dir)
        plot_correlation_heatmap(aggregated_df, args.output_dir)
        
        print("\n--- Comparison Analysis Complete ---")
        print("Generated comparison chart and correlation heatmap.")
        print("------------------------------------")

    except Exception as e:
        print(f"\nAn error occurred during comparison: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare scoring results across multiple models.")
    parser.add_argument('--scoring_base_dir', type=str, default='../result/scoring', help="Base directory where model scoring results are stored.")
    parser.add_argument('--output_dir', type=str, default='../result/comparison', help="Directory to save comparison charts and data.")
    args = parser.parse_args()
    main(args)
