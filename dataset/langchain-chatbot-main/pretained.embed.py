from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from datasets import DatasetDict, Dataset

# Define the function to filter dataset for English and Indonesian texts
# def filter_english_and_indonesian(example):
#     return example['language'] in ['en', 'id']

# dataset = load_from_disk('./pretained_dataset/local_xnli_dataset')
# print(dataset['train'][0])

# # Apply the filter function to the train, validation, and test splits
# dataset['train'] = dataset['train'].filter(filter_english_and_indonesian)
# dataset['validation'] = dataset['validation'].filter(filter_english_and_indonesian)
# dataset['test'] = dataset['test'].filter(filter_english_and_indonesian)

# Sample data (replace with your actual dataset loading logic)
data = {
    'train': {
        'text': ["This is an English sentence.", "Ini adalah kalimat dalam Bahasa Indonesia."],
        'label': [0, 1],
        'language': ['en', 'id']
    },
    'validation': {
        'text': ["An example validation sentence.", "Contoh kalimat validasi."],
        'label': [0, 1],
        'language': ['en', 'id']
    }
}

# Create a DatasetDict
dataset = DatasetDict({
    'train': Dataset.from_dict(data['train']),
    'validation': Dataset.from_dict(data['validation'])
})

# Initialize the tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

# Filter dataset for English and Indonesian (example of creating dataset from files or other sources)
def filter_english_and_indonesian(example):
    return example['language'] in ['en', 'id']

# Apply filter (if needed) - here for illustration; our dataset already filtered
dataset['train'] = dataset['train'].filter(filter_english_and_indonesian)
dataset['validation'] = dataset['validation'].filter(filter_english_and_indonesian)

# Define preprocess function
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True)

# Apply the preprocess function to the filtered datasets
encoded_dataset = dataset.map(preprocess_function, batched=True)

from transformers import TrainingArguments, Trainer

# Initialize the model
model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=3)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./pretrained",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['validation']
)

# Train and evaluate the model
trainer.train()
trainer.evaluate()