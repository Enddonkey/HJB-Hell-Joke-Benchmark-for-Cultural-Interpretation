import pandas as pd
import simpledorff
import numpy as np
from statsmodels.stats import inter_rater as irr

def preprocess_for_fleiss_kappa(df, columns):
    """
    Preprocesses the dataframe to a format suitable for statsmodels' Fleiss' Kappa calculation.
    It creates a matrix where rows are items and columns are categories,
    and values are the number of raters who assigned that category to that item.
    """
    # Melt the dataframe to long format
    long_df = df[columns].copy()
    long_df['doc_id'] = df.index
    long_df = long_df.melt(id_vars='doc_id', var_name='rater', value_name='rating')
    
    # Create the count matrix
    agg_df = long_df.groupby(['doc_id', 'rating']).size().unstack(fill_value=0)
    
    # Ensure all possible categories are present as columns, even if some were never chosen
    # This is important if some categories have 0 votes across all items
    all_categories = pd.unique(long_df['rating'].dropna())
    agg_df = agg_df.reindex(columns=all_categories, fill_value=0)
    
    return agg_df

def main():
    """
    Main function to calculate and print inter-rater agreement scores.
    """
    try:
        df = pd.read_csv('offensive - Sheet1.csv')
    except FileNotFoundError:
        print("Error: 'offensive - Sheet1.csv' not found. Please ensure the file is in the correct directory.")
        return

    # Define the columns for each dimension
    label_cols = ['label1', 'label2', 'label3']
    category_cols = ['category1', 'category2', 'category3']
    target_cols = ['target1', 'target2', 'target3']

    # --- Krippendorff's Alpha Calculation (using simpledorff) ---
    print("--- Calculating Krippendorff's Alpha ---")

    # 1. For 'label'
    # simpledorff requires a long format dataframe with columns: doc_id, rater, rating
    label_long_df = pd.melt(df.reset_index(), id_vars='index', value_vars=label_cols,
                            var_name='rater', value_name='label_rating')
    label_long_df.rename(columns={'index': 'doc_id'}, inplace=True)
    alpha_label = simpledorff.calculate_krippendorffs_alpha_for_df(label_long_df,
                                                                   experiment_col='doc_id',
                                                                   annotator_col='rater',
                                                                   class_col='label_rating')
    print(f"\nKrippendorff's Alpha for 'label': {alpha_label:.4f}")

    # 2. For 'category'
    category_long_df = pd.melt(df.reset_index(), id_vars='index', value_vars=category_cols,
                               var_name='rater', value_name='category_rating')
    category_long_df.rename(columns={'index': 'doc_id'}, inplace=True)
    alpha_category = simpledorff.calculate_krippendorffs_alpha_for_df(category_long_df,
                                                                      experiment_col='doc_id',
                                                                      annotator_col='rater',
                                                                      class_col='category_rating')
    print(f"Krippendorff's Alpha for 'category': {alpha_category:.4f}")

    # 3. For 'target'
    target_long_df = pd.melt(df.reset_index(), id_vars='index', value_vars=target_cols,
                             var_name='rater', value_name='target_rating')
    target_long_df.rename(columns={'index': 'doc_id'}, inplace=True)
    alpha_target = simpledorff.calculate_krippendorffs_alpha_for_df(target_long_df,
                                                                    experiment_col='doc_id',
                                                                    annotator_col='rater',
                                                                    class_col='target_rating')
    print(f"Krippendorff's Alpha for 'target': {alpha_target:.4f}")


    # --- Fleiss' Kappa Calculation (using statsmodels) ---
    print("\n\n--- Calculating Fleiss' Kappa ---")

    # 1. For 'label'
    # Fleiss' Kappa requires a different format: an (items x categories) matrix
    label_agg = preprocess_for_fleiss_kappa(df, label_cols)
    kappa_label = irr.fleiss_kappa(label_agg, method='fleiss')
    print(f"\nFleiss' Kappa for 'label': {kappa_label:.4f}")

    # 2. For 'category'
    category_agg = preprocess_for_fleiss_kappa(df, category_cols)
    kappa_category = irr.fleiss_kappa(category_agg, method='fleiss')
    print(f"Fleiss' Kappa for 'category': {kappa_category:.4f}")

    # 3. For 'target'
    target_agg = preprocess_for_fleiss_kappa(df, target_cols)
    kappa_target = irr.fleiss_kappa(target_agg, method='fleiss')
    print(f"Fleiss' Kappa for 'target': {kappa_target:.4f}")


if __name__ == '__main__':
    main()
