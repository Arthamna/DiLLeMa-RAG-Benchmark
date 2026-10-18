from enum import Enum


class ModelName(Enum):
  # GPT2="gpt2"
  # KOMODO="komodo"
  # MISTRAL="mistral"
  # LLAMA2="llama2"
  # SEALION2="sealion"
  GPT3_TURBO='openai'
  INDO_SENTENCE="indo_sentence"
  MINILLM_V6="minilm-v6"
  MPNET_BASE2="mpnet-base-v2"
  LABSE="labse"
  MULTILINGUAL_E5_SMALL="multilingual-e5-small"
  MULTILINGUAL_MINILM="multilingual-minilm"
  MULTILINGUAL_MINILM_FINETUNING_EARLY_STOP="multilingual-minilm-finetuning-early-stop"
  MULTILINGUAL_MINILM_FINETUNING="multilingual-minilm-finetuning"
  MULTILINGUAL_MINILM_FINETUNING_2="multilingual-minilm-finetuning-2"
  MULTILINGUAL_MINILM_FINETUNING_3="multilingual-minilm-finetuning-3"
  MULTILINGUAL_MINILM_FINETUNING_4="multilingual-minilm-finetuning-4"
  MULTILINGUAL_MINILM_FINETUNING_5="multilingual-minilm-finetuning-5"
  MULTILINGUAL_MINILM_FINETUNING_6="multilingual-minilm-finetuning-6"
  MULTILINGUAL_MINILM_FINETUNING_137="multilingual-minilm-finetuning-137"
  MULTILINGUAL_MINILM_FINETUNING_239="multilingual-minilm-finetuning-239"
  MULTILINGUAL_MINILM_FINETUNING_97_1e06="multilingual-minilm-finetuning-97-1e-06"
  MULTILINGUAL_MINILM_FINETUNING_26_1e03="multilingual-minilm-finetuning-26-1e-03"
  MULTILINGUAL_MINILM_FINETUNING_192_b8="multilingual-minilm-finetuning-192-b8"
  MULTILINGUAL_E5_SMALL_FINETUNING_1="multilingual-e5-small-finetuning-1"
  MULTILINGUAL_E5_SMALL_FINETUNING_2="multilingual-e5-small-finetuning-2"
  MULTILINGUAL_E5_SMALL_FINETUNING_3="multilingual-e5-small-finetuning-3"
  MULTILINGUAL_E5_SMALL_FINETUNING_4="multilingual-e5-small-finetuning-4"
  MULTILINGUAL_E5_SMALL_FINETUNING_5="multilingual-e5-small-finetuning-5"
  MULTILINGUAL_E5_SMALL_FINETUNING_6="multilingual-e5-small-finetuning-6"

models = {
  # ModelName.GPT2.value: {
  #   "repo_id": "openai-community/gpt2",
  #   "local_dir": "./app/model/modules/gpt2"
  # },
  # ModelName.KOMODO.value: {
  #   "repo_id": "Yellow-AI-NLP`/komodo-7b-base",
  #   "local_dir": "./app/model/modules/komodo"
  # },
  # ModelName.MISTRAL.value: {
  #   "repo_id": "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ",
  #   "local_dir": "./app/model/modules/mistral"
  # },
  # ModelName.LLAMA2.value: {
  #   "repo_id": "meta-llama/Llama-2-7b-chat-hf",
  #   "local_dir": "./app/model/modules/llama2"
  # },
  # ModelName.SEALION2.value: {
  #   "repo_id": "aisingapore/sea-lion-7b-instruct",
  #   "local_dir": "./app/model/modules/sealion"
  # },
  ModelName.INDO_SENTENCE.value: {
    "repo_id": "firqaaa/indo-sentence-bert-base",
    "local_dir": "./app/model/modules/indo-sentence-bert-base"
  },
  ModelName.MINILLM_V6.value: {
    "repo_id": "sentence-transformers/all-MiniLM-L6-v2",
    "local_dir": "./app/model/modules/all-minilm"
  },
  ModelName.MPNET_BASE2.value: {
    "repo_id": "sentence-transformers/all-mpnet-base-v2",
    "local_dir": "./app/model/modules/mpnet-base-v2"
  },
  ModelName.MULTILINGUAL_MINILM.value: {
    "repo_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "local_dir": "./app/model/modules/multilingual-minilm"
  },
  ModelName.LABSE.value: {
    "repo_id": "sentence-transformers/LaBSE",
    "local_dir": "./app/model/modules/labse"
  },
  ModelName.MULTILINGUAL_E5_SMALL.value: {
    "repo_id": "intfloat/multilingual-e5-small",
    "local_dir": "./app/model/modules/multilingual-e5-small"
  },
  ModelName.MULTILINGUAL_MINILM_FINETUNING.value: {
    "repo_id": "ramamimu/finetuning-MiniLM-L12-v2",
    "local_dir": "./app/model/modules/multilingual-minilm-finetuning"
  },
  ModelName.MULTILINGUAL_MINILM_FINETUNING_2.value: {
    "repo_id": "ramamimu/finetuning-MiniLM-L12-v2",
    "local_dir": "./app/model/modules/multilingual-minilm-finetuning-2"
  },
  ModelName.MULTILINGUAL_MINILM_FINETUNING_3.value: {
    "repo_id": "ramamimu/finetuning-MiniLM-L12-v2",
    "local_dir": "./app/model/modules/multilingual-minilm-finetuning-3"
  },
  ModelName.MULTILINGUAL_MINILM_FINETUNING_4.value: {
    "repo_id": "ramamimu/finetuning-MiniLM-L12-v2",
    "local_dir": "./app/model/modules/multilingual-minilm-finetuning-4"
  },
  ModelName.MULTILINGUAL_MINILM_FINETUNING_5.value: {
    "repo_id": "ramamimu/finetuning-MiniLM-L12-v2",
    "local_dir": "./app/model/modules/multilingual-minilm-finetuning-5"
  },
  ModelName.MULTILINGUAL_MINILM_FINETUNING_6.value: {
    "repo_id": "ramamimu/finetuning-MiniLM-L12-v2",
    "local_dir": "./app/model/modules/multilingual-minilm-finetuning-6"
  },
  ModelName.MULTILINGUAL_E5_SMALL_FINETUNING_1.value: {
    "repo_id": "intfloat/multilingual-e5-small",
    "local_dir": "./app/model/modules/multilingual-e5-small-finetuning-1"
  },
  ModelName.MULTILINGUAL_E5_SMALL_FINETUNING_2.value: {
    "repo_id": "intfloat/multilingual-e5-small",
    "local_dir": "./app/model/modules/multilingual-e5-small-finetuning-2"
  },
  ModelName.MULTILINGUAL_E5_SMALL_FINETUNING_3.value: {
    "repo_id": "intfloat/multilingual-e5-small",
    "local_dir": "./app/model/modules/multilingual-e5-small-finetuning-3"
  },
  ModelName.MULTILINGUAL_E5_SMALL_FINETUNING_4.value: {
    "repo_id": "intfloat/multilingual-e5-small",
    "local_dir": "./app/model/modules/multilingual-e5-small-finetuning-4"
  },
  ModelName.MULTILINGUAL_E5_SMALL_FINETUNING_5.value: {
    "repo_id": "intfloat/multilingual-e5-small",
    "local_dir": "./app/model/modules/multilingual-e5-small-finetuning-5"
  },
  ModelName.MULTILINGUAL_E5_SMALL_FINETUNING_6.value: {
    "repo_id": "intfloat/multilingual-e5-small",
    "local_dir": "./app/model/modules/multilingual-e5-small-finetuning-6"
  },
  ModelName.MULTILINGUAL_MINILM_FINETUNING_EARLY_STOP.value: {
    "repo_id": "ramamimu/finetuning-MiniLM-L12-v2",
    "local_dir": "./app/model/modules/multilingual-minilm-earlystop"
  },
  ModelName.MULTILINGUAL_MINILM_FINETUNING_137.value: {
    "repo_id": "ramamimu/finetuning-MiniLM-L12-v2",
    "local_dir": "./app/model/modules/multilingual-minilm-finetuning-137"
  },
  ModelName.MULTILINGUAL_MINILM_FINETUNING_239.value: {
    "repo_id": "ramamimu/finetuning-MiniLM-L12-v2",
    "local_dir": "./app/model/modules/multilingual-minilm-finetuning-239"
  },
  ModelName.MULTILINGUAL_MINILM_FINETUNING_97_1e06.value: {
    "repo_id": "ramamimu/finetuning-MiniLM-L12-v2",
    "local_dir": "./app/model/modules/multilingual-minilm-finetuning-97-1e-06"
  },
  ModelName.MULTILINGUAL_MINILM_FINETUNING_26_1e03.value: {
    "repo_id": "ramamimu/finetuning-MiniLM-L12-v2",
    "local_dir": f"./app/model/modules/{ModelName.MULTILINGUAL_MINILM_FINETUNING_26_1e03.value}"
  },
  ModelName.MULTILINGUAL_MINILM_FINETUNING_192_b8.value: {
    "repo_id": "ramamimu/finetuning-MiniLM-L12-v2",
    "local_dir": f"./app/model/modules/{ModelName.MULTILINGUAL_MINILM_FINETUNING_192_b8.value}"
  },
}

# labse berat
# nomic failed to fulfill evaluation due to run out of GPU memory in IndonesianMongabayConservationClassification task

dataset_iftegration = [
  # {
  #   "folder": "international exposure id",
  #   "file": "international expo.csv"
  # },
  # {
  #   "folder": "international exposure en",
  #   "file": "international expo.csv"
  # },
  # {
  #   "folder": "akademik",
  #   "file": "akademik.csv"
  # },
  # {
  #   "folder": "program internasional",
  #   "file": "program internasional.csv"
  # },
  # {
  #   "folder": "akademik luar kampus",
  #   "file": "akademik luar kampus.csv"
  # },
  # {
  #   "folder": "yudisium",
  #   "file": "yudisium.csv"
  # },
  # {
  #   "folder": "dana perkuliahan",
  #   "file": "dana perkuliahan.csv"
  # },
  {
    "folder": "akademik s1",
    "file": "Akademik S1.csv"
  },
  # {
  #   "folder": "akademik s2",
  #   "file": "Akademik S2.csv"
  # },
  # {
  #   "folder": "beasiswa",
  #   "file": "beasiswa.csv"
  # },
  # {
  #   "folder": "dana pendidikan",
  #   "file": "dana pendidikan.csv"
  # },
  # {
  #   "folder": "kerja praktik",
  #   "file": "kerja praktik.csv"
  # },
  # {
  #   "folder": "magang",
  #   "file": "magang.csv"
  # },
  # {
  #   "folder": "MBKM",
  #   "file": "MBKM.csv"
  # },
  # {
  #   "folder": "SKEM",
  #   "file": "SKEM.csv"
  # },
  # {
  #   "folder": "tesis",
  #   "file": "tesis.csv"
  # },  
  # {
  #   "folder": "wisuda",
  #   "file": "wisuda.csv"
  # },  
  # {
  #   "folder": "yudisium dan tugas akhir",
  #   "file": "yudisium dan tugas akhir.csv"
  # },
]
