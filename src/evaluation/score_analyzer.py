import os
import re
import json
import argparse
import pandas as pd
import numpy as np
from openai import OpenAI
from tqdm import tqdm
import concurrent.futures
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
MAX_WORKERS = 10

# --- OpenAI Client Initialization ---
def get_openai_client(config_path):
    config = {}
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            for line in f:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    config[key.strip()] = value.strip()
        api_key = config.get('OPENAI_API_KEY')
        base_url = config.get('OPENAI_BASE_URL')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in config file.")
        print(f"Initializing scoring client with base URL: {base_url}")
        return OpenAI(api_key=api_key, base_url=base_url)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error initializing OpenAI client: {e}")
        return None

# --- Data Loading and Merging ---
def load_and_merge_data(model_eval_path, true_jokes_path):
    # 1. Load detailed classification results
    detailed_results_file = os.path.join(model_eval_path, 'detailed_results.csv')
    if not os.path.exists(detailed_results_file):
        raise FileNotFoundError(f"Detailed results file not found: {detailed_results_file}")
    df_class = pd.read_csv(detailed_results_file)
    df_class['id'] = df_class['id'].astype(str)

    # 2. Load analyzed jokes (for background/logic predictions)
    analyzed_jokes_file = os.path.join(model_eval_path, 'analyzed_jokes.jsonl')
    if not os.path.exists(analyzed_jokes_file):
        raise FileNotFoundError(f"Analyzed jokes file not found: {analyzed_jokes_file}")
    
    analyzed_jokes_data = []
    with open(analyzed_jokes_file, 'r', encoding='utf-8') as f:
        for line in f:
            analyzed_jokes_data.append(json.loads(line))
    df_analyzed = pd.DataFrame(analyzed_jokes_data)
    if 'content' in df_analyzed.columns:
        df_analyzed = df_analyzed.drop(columns=['content'])
    df_analyzed = df_analyzed.rename(columns={'Background': 'pred_background', 'Logic': 'pred_logic'})
    df_analyzed['id'] = df_analyzed['id'].astype(str)

    # 3. Load true joke explanations
    if not os.path.exists(true_jokes_path):
        raise FileNotFoundError(f"True jokes file not found: {true_jokes_path}")
    
    true_jokes_data = []
    with open(true_jokes_path, 'r', encoding='utf-8') as f:
        for line in f:
            true_jokes_data.append(json.loads(line))
    df_true_exp = pd.DataFrame(true_jokes_data)
    if 'content' in df_true_exp.columns:
        df_true_exp = df_true_exp.drop(columns=['content'])
    df_true_exp = df_true_exp.rename(columns={'Background': 'true_background', 'Logic': 'true_logic'})
    df_true_exp['id'] = df_true_exp['id'].astype(str)

    # Merge all dataframes
    df_merged = pd.merge(df_class, df_analyzed, on='id', how='left')
    df_merged = pd.merge(df_merged, df_true_exp, on='id', how='left')
    
    # Final check for missing data
    missing_rows = df_merged['pred_background'].isnull().sum() + df_merged['true_background'].isnull().sum()
    if missing_rows > 0:
        print(f"Warning: Found {missing_rows} rows with missing background/logic data after merge. This may affect scoring.")

    return df_merged

# --- Scoring ---
def calculate_classification_scores(df):
    print("Calculating classification scores (Label, Category, Target)...")
    def score_label(row):
        try:
            true_label, pred_label = int(row['true_label']), int(row['pred_label'])
            diff = abs(true_label - pred_label)
            return 3 if diff == 0 else 1 if diff == 1 else 0
        except (ValueError, TypeError):
            return 0
    df['label_score'] = df.apply(score_label, axis=1)
    df['category_score'] = (df['true_category'] == df['pred_category']).astype(int)
    df['target_score'] = (df['true_target'] == df['pred_target']).astype(int)
    print("Classification scores calculated.")
    return df

def create_explanation_scoring_prompt(joke_content, true_background, true_logic, pred_background, pred_logic):
    return f"""
    **任务：评估模型对笑话背后逻辑的理解程度**
    请根据下方提供的“真实解释”和“模型预测”，以及详细的评分标准，对模型在“背景（background）”和“逻辑（logic）”两个维度上的表现进行评分。
    **笑话内容:**
    "{joke_content}"
    ---
    **真实解释:**
    - **背景（Background）:** {true_background}
    - **逻辑（Logic）:** {true_logic}
    **模型预测:**
    - **背景（Background）:** {pred_background}
    - **逻辑（Logic）:** {pred_logic}
    ---
    **评分标准:**
    **1. 背景（Background）理解程度评分:**
    - **3分（完全理解）:** 能准确说出梗绑定的核心历史/社会事件。
    - **2分（部分理解）:** 能提及相关领域但不够精确，或只说出部分关键信息。
    - **0分（未理解）:** 完全答非所问，无法关联任何相关背景。
    **2. 逻辑（Logic）理解程度评分:**
    - **3分（完全理解）:** 能准确识别并解释笑点核心的反差逻辑（如：严肃背景与戏谑表达的反差、语言双关等）。
    - **2分（部分理解）:** 仅能说出表面逻辑（如事件本身），但未识别或解释其幽默反差的核心。
    - **0分（未理解）:** 无法梳理逻辑，甚至误解笑点。
    ---
    **输出要求:**
    你的回答必须是一个单独且有效的JSON对象，包含 "background_score" 和 "logic_score" 两个键，其值为整数（0, 2, 或 3）。
    **输出示例:**
    {{
      "background_score": 3,
      "logic_score": 2
    }}
    """

def get_explanation_scores(client, prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        return {"background_score": data.get("background_score"), "logic_score": data.get("logic_score")}
    except Exception:
        return {"background_score": np.nan, "logic_score": np.nan}

def score_explanations_with_llm(df, client):
    print("Scoring explanations (Background, Logic) using LLM...")
    tasks = [(row['id'], create_explanation_scoring_prompt(row['content'], row.get('true_background', ''), row.get('true_logic', ''), row.get('pred_background', ''), row.get('pred_logic', ''))) for _, row in df.iterrows()]
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_id = {executor.submit(get_explanation_scores, client, task[1]): task[0] for task in tasks}
        for future in tqdm(concurrent.futures.as_completed(future_to_id), total=len(tasks), desc="Scoring Explanations"):
            joke_id = future_to_id[future]
            try:
                results[joke_id] = future.result()
            except Exception as e:
                print(f"Joke ID {joke_id} generated an exception: {e}")
                results[joke_id] = {"background_score": np.nan, "logic_score": np.nan}
    score_df = pd.DataFrame.from_dict(results, orient='index').reset_index().rename(columns={'index': 'id'})
    score_df['id'] = score_df['id'].astype(str)
    df = pd.merge(df, score_df, on='id', how='left')
    print("Explanation scores calculated.")
    df[['background_score', 'logic_score']] = df[['background_score', 'logic_score']].fillna(0).astype(int)
    return df

# --- Analysis and Visualization ---
def analyze_scores_and_find_worst_examples(df):
    print("Analyzing scores and finding worst examples...")
    score_columns = ['label_score', 'category_score', 'target_score', 'background_score', 'logic_score']
    summary = {'average_scores': {}, 'worst_examples': {}}
    for col in score_columns:
        summary['average_scores'][col] = df[col].mean()
    df['total_score'] = df[score_columns].sum(axis=1)
    summary['average_scores']['total_score'] = df['total_score'].mean()
    for col in score_columns:
        worst_idx = df[col].idxmin()
        worst_example = df.loc[worst_idx].to_dict()
        for key, value in worst_example.items():
            if isinstance(value, (np.integer, np.floating, np.bool_)):
                worst_example[key] = value.item()
        summary['worst_examples'][col] = worst_example
    print("Analysis complete.")
    return summary

def plot_scores(summary, model_name, output_dir):
    print("Generating score visualization...")
    avg_scores = summary['average_scores']
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(avg_scores.keys()), y=list(avg_scores.values()), palette='viridis', hue=list(avg_scores.keys()), legend=False)
    plt.title(f'Average Scores for {model_name}', fontsize=16)
    plt.ylabel('Average Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max(3.0, max(avg_scores.values()) * 1.1))
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{model_name}_scores_visualization.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Score visualization saved to {plot_path}")

# --- Main Execution ---
def main(args):
    output_dir = os.path.join(args.output_base_dir, args.model_name)
    os.makedirs(output_dir, exist_ok=True)
    try:
        full_df = load_and_merge_data(os.path.join(args.evaluation_base_dir, args.model_name), args.true_jokes_file)
        full_df = calculate_classification_scores(full_df)
        client = get_openai_client(args.api_key_path)
        if not client:
            print("Failed to initialize OpenAI client. Aborting.")
            return
        full_df = score_explanations_with_llm(full_df, client)
        summary_data = analyze_scores_and_find_worst_examples(full_df)
        final_csv_path = os.path.join(output_dir, 'detailed_scores.csv')
        full_df.to_csv(final_csv_path, index=False, encoding='utf-8-sig')
        print(f"Detailed scoring results saved to {final_csv_path}")
        summary_json_path = os.path.join(output_dir, 'scoring_summary.json')
        with open(summary_json_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=4, ensure_ascii=False)
        print(f"Scoring summary saved to {summary_json_path}")
        plot_scores(summary_data, args.model_name, output_dir)
        print("\n--- Scoring Analysis Summary ---")
        for key, value in summary_data['average_scores'].items():
            print(f"Average {key}: {value:.4f}")
        print("---------------------------------")
    except (FileNotFoundError, ValueError) as e:
        print(f"\nAn error occurred: {e}")
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Score and analyze a model's performance on joke classification and explanation.")
    parser.add_argument('--model_name', type=str, required=True, help="Name of the model to score.")
    parser.add_argument('--evaluation_base_dir', type=str, default='../result/evaluation', help="Base directory for evaluation results.")
    parser.add_argument('--true_jokes_file', type=str, default='../data/label_jokes.jsonl', help="Path to the true jokes JSONL file.")
    parser.add_argument('--output_base_dir', type=str, default='../result/scoring', help="Base directory to save scoring results.")
    parser.add_argument('--api_key_path', type=str, default='../openai_key.txt', help="Path to the OpenAI config file.")
    args = parser.parse_args()
    main(args)
