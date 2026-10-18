import pandas as pd
import jsonlines

# Load the CSV file
csv_file = 'fine_tuning_dataset/all_merged_dataset.csv'  # Replace with your CSV file path
data = pd.read_csv(csv_file)

# Check the first few rows to ensure it's loaded correctly
print(data.head())

# Function to convert each row into the JSONL conversation format
def convert_to_jsonl(row):
    return {
        "messages": [
            {
                "role": "system", 
                "content": "Anda adalah chatbot interaktif bernama Tanyabot untuk menjawab pertanyaan seputar akademik departemen Teknik Informatika, Institut Teknologi Sepuluh Nopember Surabaya."
            },
            {
                "role": "user", 
                "content": f"{row['question']}"
            },
            {
                "role": "assistant", 
                "content": f"{row['context']}"  # Replace <ANSWER> with the actual answer if available
            }
        ]
    }

# Apply function to each row and save to JSONL
jsonl_file = 'output_file.jsonl'  # Replace with your desired output file path
with jsonlines.open(jsonl_file, mode='w') as writer:
    for _, row in data.iterrows():
        writer.write(convert_to_jsonl(row))

print(f"Conversion completed! JSONL file saved as {jsonl_file}")