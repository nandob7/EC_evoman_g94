import pandas as pd
import glob
import os

# Define the folder path, file pattern, and columns to remove
folder_path = 'runs/generalist'  # Replace with the path to your folder
file_pattern = '**/all_statistics*.csv'  # The '**' makes the search recursive
columns_to_remove = ['Best Genome']  # Replace with the column names you want to remove

# Get all the CSV files matching the pattern (including subfolders)
csv_files = glob.glob(os.path.join(folder_path, file_pattern), recursive=True)

# Iterate over each CSV file
for file in csv_files:
    # Read the CSV file
    df = pd.read_csv(file)

    # Remove the specified columns, ignoring errors if the columns don't exist
    df.drop(columns=columns_to_remove, inplace=True, errors='ignore')

    # Save the modified DataFrame back to the CSV
    df.to_csv(file, index=False)

    print(f"Processed {file}")
