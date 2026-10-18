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

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_openai import ChatOpenAI

chat = ChatOpenAI(streaming=True)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="context"),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ChatMessageHistory


chain = prompt | chat | StrOutputParser()
question = "bagaimana cara mendaftar PKM-KI?"
ctx = retriever.get_relevant_documents(question)

message_history = ChatMessageHistory()

message_history.add_user_message("Hi, saya Rama")
message_history.add_ai_message("Halo, saya tanyabot")
message_history.add_user_message("Siapa nama saya")
message_history.add_ai_message("Anda adalah Rama?")
message_history.add_user_message(question)


# print(
#   chain.invoke(
#       {
#           "context": [HumanMessage(content=format_docs(ctx))],
#           "messages": message_history.messages,
#       }
#   )
# )

text = ""
for chunk in chain.stream(
  {
    "context": [HumanMessage(content=format_docs(ctx))],
    "messages": message_history.messages,
  }
):
      if chunk:
        content = chunk.replace("\n", "")
        text += content
        print("\n")
        print(text)

message_history.add_ai_message(text)
print(message_history.messages)
print(len(message_history.messages))

# second question
message_history.add_user_message("what i ask before?")
text = ""
for chunk in chain.stream(
  {
    "context": [HumanMessage(content=format_docs(ctx))],
    "messages": message_history.messages,
  }
):
      if chunk:
        content = chunk.replace("\n", "")
        text += content

print("\n=========== second question ===========")
print(text)
