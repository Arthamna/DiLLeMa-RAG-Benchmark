import os
import pandas as pd

# Folder path containing the CSV files
folder_path = "fine_tuning_dataset"

# List to hold DataFrames
dfs = []

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        # Read each CSV file and append it to the list
        print(file_path)
        df = pd.read_csv(file_path)
        dfs.append(df)

# Concatenate all DataFrames in the list into a single DataFrame
merged_df = pd.concat(dfs, ignore_index=True)

# Save the merged DataFrame to a new CSV file
output_file = os.path.join(folder_path, "all_merged_dataset.csv")
merged_df.to_csv(output_file, index=False)

print(f"Merged dataset saved to {output_file}")