from enum import Enum
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

# get embedding
# load all local embedded texts
from langchain_community.vectorstores import FAISS
from app.model import load_model
from langchain.embeddings.openai import OpenAIEmbeddings
from models import models, ModelName, dataset_iftegration

def get_embedding_model(model_name: str = ModelName.GPT3_TURBO.value):
  if model_name == ModelName.GPT3_TURBO.value:
    embedding_model = OpenAIEmbeddings()
    fvs_init_path="./app/embeddings/openai"
    return embedding_model, fvs_init_path
  
  embedding_model = load_model(models[model_name]['local_dir'])
  fvs_init_path=f"./app/embeddings/{model_name}"
  return embedding_model, fvs_init_path

def get_vectorstore(embedding_model, fvs_init_path, df_dataset):
  faiss = None

  done = []
  for _, row in df_dataset.iterrows():
    title = row['Judul']
    print(title)

    if title not in done:
      folder_path=f"{fvs_init_path}/{title}"
      if not faiss:
        faiss = FAISS.load_local(folder_path=folder_path, embeddings=embedding_model, allow_dangerous_deserialization=True)
      else:
        temp_faiss = FAISS.load_local(folder_path=folder_path, embeddings=embedding_model, allow_dangerous_deserialization=True)
        faiss.merge_from(temp_faiss)
      done.append(title)
    else:
      print(">>>>> skipped")

  return faiss

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# prompt
template = """SYSTEM: Anda adalah chatbot interaktif bernama TANYABOT, jawablah pertanyaan secara detail dari konteks yang diberikan. Jika Anda tidak mengetahui jawabannya, katakan "saya tidak tahu".
  jawablah pertanyaan menggunakan Bahasa Indonsia.
  CONTEXT: {context}

  Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Chain
# use case
# chain.invoke({"context": doc, "question": "tahun berapa dokumen dibuat?"})
from langchain_core.output_parsers import StrOutputParser
chain = prompt | llm | StrOutputParser()

# get question and answer from dataframe
def get_question_ground_truth_from_dataset(df_dataset):
  questions = []
  expected_answers = []

  for _, row in df_dataset.iterrows():
    question = row['Question']
    expected_answer = row['Expected Answer']

    questions.append(question)
    expected_answers.append(expected_answer)
  
  return questions, expected_answers

# get context and generated answer from dataframe
def get_context_answer(questions, vestorstore, k=3):
  print("========= CHAIN CALLED =========")
  all_context = []
  all_answer = []
  for question in questions:
    contexts = []

    vs_result = vestorstore.similarity_search(question, k=k)
    for item in vs_result:
      contexts.append(item.page_content)

    all_context.append(contexts)
    all_answer.append(chain.invoke({"context": vs_result, "question": question}))
  return all_context, all_answer

from datasets import Dataset 
# import metrics
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas import evaluate
def evaluate(questions, ground_truth, contexts, answers, file_to_save_path):
  print("================= START EVALUATE =================")
  data_samples_s1 = {
    'question': questions,
    "ground_truth": ground_truth,
    "contexts": contexts,
    "answer": answers
  }

  dataset = Dataset.from_dict(data_samples_s1)

  result = evaluate(
      dataset,
      metrics=[
          context_precision,
          faithfulness,
          answer_relevancy,
          context_recall,
      ],
  )

  # result
  df_evaluation_s1 = result.to_pandas()
  print(df_evaluation_s1.head())
  df_evaluation_s1.to_csv(file_to_save_path)

def calculate_context(df_dataset_path, df_qa_path, embedding_model_name, save_local_path, k=3):
  df_dataset = pd.read_csv(df_dataset_path)
  df_qa = pd.read_csv(df_qa_path)
  embedding_model, model_path = get_embedding_model(embedding_model_name)
  vectorstore = get_vectorstore(embedding_model, model_path, df_dataset)
  questions, expected_answers = get_question_ground_truth_from_dataset(df_qa)
  contexts, answers = get_context_answer(questions, vectorstore, k=k)
  df_contexts = pd.DataFrame({
    'question': questions,
    "ground_truth": expected_answers,
    "contexts": contexts,
    "answer": answers
  })
  df_contexts.to_csv(save_local_path)
  print(f"{df_dataset_path} saved into {save_local_path}")

model_names = [
  ModelName.INDO_SENTENCE.value
]

# default 4
k_params = [
  5, 6, 7
]

# embedding_model_name = ModelName.GPT3_TURBO.value
for embedding_model_name in model_names:
  print(f"============ Populate {embedding_model_name} ============")
  for item in dataset_iftegration:
    for k in k_params:
      print(f"============ {item} - k param {k} ============")
      calculate_context(
        f'dataset/{item["folder"]}/Dataset IF-Tegration - {item["file"]}',
        f'dataset/{item["folder"]}/QA Generator - {item["file"]}',
        embedding_model_name,
        f'dataset/{item["folder"]}/evaluation context {embedding_model_name}-{k}.csv',
        k=k
      )