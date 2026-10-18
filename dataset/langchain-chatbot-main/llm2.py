import datetime
import os
from dotenv import load_dotenv

from app.embeddings import get_embeddings
from langchain_openai import ChatOpenAI

from huggingface_hub import login
import torch

load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
login(HUGGINGFACEHUB_API_TOKEN)

# ============= helper function ============= # 
def format_docs(docs):
  return "\n\n".join(doc.page_content for doc in docs)

def print_response(response):
  print("\n============ Content ============")
  print("RAW:", response)
  print("============")
  print("question:", response["question"])
  print("answer:", response["answer"])

# ==================== ## ==================== # 

vectorstore = get_embeddings("app/embeddings")
retriever=vectorstore.as_retriever(search_kwargs={"k": 1})


from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

print("start init: ", datetime.datetime.now())
# Chain
def get_stream_chain():
  return (
  {"context": retriever, "question": RunnablePassthrough()}
  | prompt 
  | llm 
  | StrOutputParser()
)
chain = get_stream_chain()
print("finish init: ", datetime.datetime.now())


# ==================== ## ==================== # 
question="tahun berapa dokumen dibuat?"
doc=retriever.get_relevant_documents(question)

# Run
for chunk in chain.stream(question):
        content = chunk.replace("\n", "<br>")
        print(chunk)

# print(chain.stream(question))