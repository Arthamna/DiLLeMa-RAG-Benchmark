import json

# Load the dataset
with open('finetuning_result/train_dataset.json', 'r', encoding='utf8') as file:
    dataset = json.load(file)

# Snapshot of dataset
for entry in dataset:
    print(f"Question: {entry['anchor']}")
    print(f"Answer: {entry['positive']}")
    print(f"ID: {entry['id']}\n")