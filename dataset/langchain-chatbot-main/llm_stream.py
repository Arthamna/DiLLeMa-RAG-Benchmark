import os
from dotenv import load_dotenv
import asyncio

from app.embeddings import get_embeddings
from app.llm import get_chain_context_huggingface, gptq, tokenizer

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

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

models = {
  "indo_sentence": {
    "repo_id": "firqaaa/indo-sentence-bert-base",
    "local_dir": "./app/model/modules/indo-sentence-bert-base"
  },
  "gpt2": {
    "repo_id": "openai-community/gpt2",
    "local_dir": "./app/model/modules/gpt2"
  },
  "komodo": {
    "repo_id": "Yellow-AI-NLP/komodo-7b-base",
    "local_dir": "./app/model/modules/komodo"
  },
  "mistral": {
    "repo_id": "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ",
    "local_dir": "./app/model/modules/mistral"
  },
  "llama2": {
    "repo_id": "meta-llama/Llama-2-7b-chat-hf",
    "local_dir": "./app/model/modules/llama2"
  },
  "sealion": {
    "repo_id": "aisingapore/sea-lion-7b-instruct",
    "local_dir": "./app/model/modules/sealion"
  }
}

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
def get_llm(name:str = "openai", model = "mistral"):
  if name == "openai":
    return ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
  else:
    path = models[model]["local_dir"]
    token = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=path)

    model_gptq = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=path,
        device_map="auto",
        torch_dtype=torch.float16
        )

    return get_chain_context_huggingface(model_gptq, token)

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, ai_prefix="tanyabot")
store={}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

from langchain_core.output_parsers import StrOutputParser

def create_stream_chain():
  return  ConversationalRetrievalChain.from_llm(
    llm=get_llm(),
    retriever=vectorstore.as_retriever(),
    memory=memory,
) 
# | StrOutputParser()

conversation_chain = create_stream_chain()

# solver: https://medium.com/llm-projects/langchain-openai-streaming-101-in-python-edd60e84c9ca

def ask_question(question):
  #  asynchronous generator
  try:
    for chunk in conversation_chain.stream({'question':question}):
      print("masuk")
      print(chunk, end="", flush=True)
  except:
    print("\n\n==============\n end of stream")

ask_question("what is PKM KI")
# ask_question("tell me a story")
# ask_question("apa itu AI")

