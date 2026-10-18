from enum import Enum

prefix = "./app/model/modules/"

class GenerationModel(Enum):
  GPT3_5 = "gpt-3.5-turbo"
  MISTRAL7B = "mistralai/Mistral-7B-Instruct-v0.3"
  LLAMA2 = "meta-llama/Llama-2-7b-chat-hf"
  AYA8B = "CohereForAI/aya-23-8B"
  SEALION7B_INSTRUCT = "aisingapore/sea-lion-7b-instruct"
  BLOOM = "bigscience/bloom-560m"
  MERAK7B_ICHSAN = "Ichsan2895/Merak-7B-v5-PROTOTYPE1-GGUF"
  MERAK7B_ASYAFIQ = "asyafiqe/Merak-7B-v3-Mini-Orca-Indo-GGUF"
  KOMODO = "Yellow-AI-NLP/komodo-7b-base"