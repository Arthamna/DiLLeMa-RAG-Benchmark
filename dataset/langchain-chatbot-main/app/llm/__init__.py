from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline
import torch
from transformers import logging
import transformers
from langchain import HuggingFacePipeline, PromptTemplate, LLMChain

logging.set_verbosity_warning()

# Some error in colab. fix with
import locale
locale.getpreferredencoding = lambda: "UTF-8"

# release cuda cache not used
torch.cuda.empty_cache()

def gptq(model_id):
  model_gptq = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        )
  return model_gptq

def tokenizer(model_id):
  tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
  return tokenizer

def get_chain_context_huggingface(model_gptq, tokenizer):
  ppline = pipeline(
          "text-generation",
          model=model_gptq,
          tokenizer=tokenizer,
          use_cache=True,
          device_map="auto",
          max_new_tokens=296, #296
          temperature=0.1,
          return_full_text=False,
          do_sample=True,
          top_k=10,
          num_return_sequences=1,
          eos_token_id=tokenizer.eos_token_id,
          pad_token_id=tokenizer.eos_token_id,
  )
  llm = HuggingFacePipeline(pipeline=ppline)
  return llm
  # prompt_context = PromptTemplate.from_template(
  #   """
  #   <s> [INST]Anda adalah asisten untuk tugas menjawab pertanyaan. Hanya gunakan potongan konteks yang diambil untuk menjawab pertanyaan. Jika tidak ada jawaban dari konteks, katakan "Saya tidak tahu". Gunakan maksimal dua kalimat dan buatlah jawaban yang ringkas.[/INST] </s>
  #   [INST]Context: {context}
  #   Pertanyaan: {question}
  #   Jawaban: [/INST]
  #   """.strip()
  # )
  # llm_chain_context = LLMChain(prompt=prompt_context ,llm=llm)
  # return llm_chain_context
