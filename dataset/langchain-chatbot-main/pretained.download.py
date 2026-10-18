from datasets import load_dataset

# Load dataset with 'all_languages' configuration
dataset = load_dataset('xnli', 'all_languages')
dataset.save_to_disk('./pretained_dataset/local_xnli_dataset')
