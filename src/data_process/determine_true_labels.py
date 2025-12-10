import pandas as pd

def determine_true_label(row, columns, tie_breaker_column):
    """
    Determines the true label for a set of columns using majority vote.
    If there is a tie, the value from the tie_breaker_column is used.
    """
    # Extract the values from the specified columns for the current row
    values = row[columns]
    
    # Calculate the mode using pandas Series' mode method
    mode_result = values.mode()
    
    # Check for a tie. A tie occurs if there are multiple modes,
    # which for 3 annotators means all three chose different options.
    # A tie can also happen if mode() returns more than one value.
    if len(mode_result) != 1:
        # This is a tie (e.g., [1, 2, 3] or ['A', 'B', 'C'])
        return row[tie_breaker_column]
    else:
        # No tie, return the first (and only) mode
        return mode_result[0]

def main():
    """
    Main function to read the data, determine true labels, and save the result.
    """
    # Define file paths
    input_file = 'data/offensive - Sheet1.csv'
    output_file = 'data/offensive_with_true_labels.csv'

    print(f"Reading data from '{input_file}'...")
    try:
        df = pd.read_csv(input_file)
        # Clean up unnamed columns that might exist in the source CSV
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return

    print("Determining true labels using majority vote...")

    # Define the column groups for voting
    label_cols = ['label1', 'label2', 'label3']
    category_cols = ['category1', 'category2', 'category3']
    target_cols = ['target1', 'target2', 'target3']

    # Apply the voting logic to each dimension (label, category, target)
    df['true_label'] = df.apply(determine_true_label, axis=1, columns=label_cols, tie_breaker_column='label2')
    df['true_category'] = df.apply(determine_true_label, axis=1, columns=category_cols, tie_breaker_column='category2')
    df['true_target'] = df.apply(determine_true_label, axis=1, columns=target_cols, tie_breaker_column='target2')

    # Reorder columns to place the new 'true' columns after 'target3'
    try:
        # Find the index of the 'target3' column
        target3_index = df.columns.get_loc('target3')
        
        # Create the new column order
        original_cols = list(df.columns)
        # Remove the new columns from their current position (end of the list)
        original_cols.remove('true_label')
        original_cols.remove('true_category')
        original_cols.remove('true_target')
        
        # Insert the new columns after 'target3'
        new_order = original_cols[:target3_index + 1] + ['true_label', 'true_category', 'true_target'] + original_cols[target3_index + 1:]
        df = df[new_order]
    except KeyError:
        print("Warning: 'target3' column not found. Appending new columns to the end.")


    print(f"Saving the results with true labels to '{output_file}'...")
    try:
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"Successfully created '{output_file}' with true labels.")
        print("\nPreview of the first 5 rows with reordered columns:")
        print(df.head())
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")

if __name__ == '__main__':
    main()
