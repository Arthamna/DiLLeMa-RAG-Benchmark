import csv
import os
import re

def list_files_in_directory(directory_path):
    # List all files in the provided directory
    files = os.listdir(directory_path)
    return files

def extract_filename(file_name):
    # Define the regex pattern to match the desired part of the file name
    pattern = r" - (.+)\.csv"
    
    # Search for the pattern in the file name
    match = re.search(pattern, file_name)
    
    # If a match is found, return the captured group
    if match:
        return match.group(1)
    else:
        return None
                            
def csv_to_txt(csv_file_path, txt_file_path, column_name):
    # Open the CSV file
    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        # Open the TXT file in write mode
        with open(txt_file_path, mode='w', encoding='utf-8') as txt_file:
            # Iterate through each row in the CSV
            for row in csv_reader:
                # Write the content of the specified column to the TXT file
                txt_file.write(row[column_name] + '\n')

def create_directory(directory_path):
    # Check if the directory already exists
    if not os.path.exists(directory_path):
        os.mkdir(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")

# Example usage
# csv_file_path = 'dataset/per files dataset/Dataset IF-Tegration - Academic Support.csv'  # Replace with your CSV file path
# txt_file_path = 'dataset/txt files/output.txt'  # Replace with your desired TXT file path
column_name = 'konten'  # Name of the column to be extracted

# csv_to_txt(csv_file_path, txt_file_path, column_name)
# print(list_files_in_directory("dataset/per files dataset"))
dir_dataset = "dataset/per files dataset"
files = list_files_in_directory(dir_dataset)
for file in files:
    filename = extract_filename(file)
    dir_txt_name = f"dataset/txt files/preprocessing-{filename}"
    create_directory(dir_txt_name)
    csv_to_txt(f"{dir_dataset}/{file}", f"{dir_txt_name}/{filename}.txt", column_name)