from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset 
# import metrics
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas import evaluate
import pandas as pd
import ast
from models import models, ModelName, dataset_iftegration

def evaluate_result(context_path: str, file_to_save_path: str):
  print("================= START EVALUATE =================")
  print(f"calculating {context_path}")
  df_context = pd.read_csv(context_path)
  df_context['contexts'] = df_context['contexts'].apply(ast.literal_eval)


  data_sample = {
    'question': df_context['question'].to_list(),
    'answer': df_context['answer'].to_list(),
    'contexts': df_context['contexts'].to_list(),
    'ground_truth': df_context['ground_truth'].to_list(),
  }
  dataset = Dataset.from_dict(data_sample)

  result = evaluate(
      dataset,
      metrics=[
          context_precision,
          faithfulness,
          answer_relevancy,
          context_recall,
      ],
      raise_exceptions=False
  )

  # result
  df_evaluation_s1 = result.to_pandas()
  print(df_evaluation_s1.head())
  df_evaluation_s1.to_csv(file_to_save_path)
  print(f"success collecting {context_path}")
  print("================= FINISH EVALUATE =================")

model_names = [
  ModelName.INDO_SENTENCE.value
]

k_params = [
  5, 6, 7
]

for model_name in model_names:
  for i in dataset_iftegration:
    for k in k_params:
      if k == 4:
        evaluate_result(f'dataset/{i["folder"]}/evaluation context {model_name}.csv', f'dataset/{i["folder"]}/evaluation result {model_name}.csv')
      else:
        evaluate_result(f'dataset/{i["folder"]}/evaluation context {model_name}-{k}.csv', 
                        f'dataset/{i["folder"]}/evaluation result {model_name}-{k}.csv')
