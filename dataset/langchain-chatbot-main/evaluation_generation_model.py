from models_generation import GenerationModel, prefix

from langchain_openai import ChatOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline
import torch
from langchain import HuggingFacePipeline

def get_llm(model_id: str):
  if model_id == GenerationModel.GPT3_5.value:
    return ChatOpenAI()
  
  path_dir = f"{prefix}{model_id}"
  token = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=path_dir)
  model_gptq = AutoModelForCausalLM.from_pretrained(
      pretrained_model_name_or_path=path_dir,
      device_map="auto",
      torch_dtype=torch.float16
      )
  
  ppline = pipeline(
    "text-generation",
    model=model_gptq,
    tokenizer=token,
    device_map="auto",
    max_new_tokens=296, #296
    temperature=0.1,
    return_full_text=False,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=token.eos_token_id,
    pad_token_id=token.eos_token_id
  )
  
  llm = HuggingFacePipeline(pipeline=ppline)
  return llm

from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from time import time

def get_chain(model_id: str):
  system_msg = (
      "Anda adalah chatbot interaktif bernama Tanyabot. Anda bertugas untuk menjawab seputar Akademik Departemen Teknik Informatika ITS\n"
      "Ikuti instruksi ini untuk menjawab pertanyaan/question: jawablah pertanyaan/question dari context yang telah diberikan. Berikan jawaban yang relevan, jika Anda tidak berhasil mendapatkan jawaban, katakan 'saya tidak tahu'.\n"
    )

  prompt = ChatPromptTemplate.from_messages(
      [
        (
          "system",
          system_msg
        ),
        MessagesPlaceholder(variable_name="context"),
        MessagesPlaceholder(variable_name="messages")
      ]
    )
  
  chain = (
    prompt
    | get_llm(model_id)
    | StrOutputParser()
    )
  
  return chain

context = """
Candi Borobudur, yang terletak di Magelang, Jawa Tengah, Indonesia, merupakan sebuah monumen megah yang melambangkan kekayaan warisan budaya daerah tersebut. Dibangun pada abad ke-9 oleh Dinasti Sailendra, bangunan ini tidak hanya berfungsi sebagai tempat ibadah, melainkan juga sebagai keajaiban arsitektur yang memukau. Berdiri elegan di puncak bukit, candi ini dikelilingi oleh pepohonan hijau yang menciptakan suasana damai. Keindahan struktur ini tercermin dalam perpaduan unik antara arsitektur Indonesia dan seni India. Candi ini memiliki sembilan platform bertingkat, di setiap tingkatnya dihiasi dengan relief batu yang kompleks, menggambarkan ajaran Buddha dan cerita rakyat Jawa. Kompleks ini terdiri dari enam teras berbentuk bujur sangkar dengan tiga pelataran melingkar, di dindingnya dihiasi dengan 2.672 panel relief, dan aslinya menyertakan 504 arca Buddha. Stupa utama terbesar berada di tengah, memahkotai bangunan ini, dan dikelilingi oleh tiga barisan melingkar 72 stupa berlubang yang menampung arca Buddha tengah duduk bersila dalam posisi teratai sempurna dengan mudra Dharmachakra (memutar roda dharma). Sejarah mencatat bahwa Borobudur ditinggalkan pada abad ke-10 ketika pusat Kerajaan Mataram Kuno dipindahkan ke Jawa Timur oleh Pu Sindok. Kesadaran dunia terhadap keberadaan candi ini muncul setelah penemuan pada tahun 1814 oleh Sir Thomas Stamford Raffles, yang saat itu menjabat sebagai Gubernur Jenderal Inggris di Jawa. Hingga kini, Borobudur masih menjadi tempat ziarah keagamaan, di mana setiap tahun umat Buddha dari seluruh Indonesia dan mancanegara berkumpul untuk memperingati Trisuci Waisak.
Candi Borobudur, yang terletak di Magelang, Jawa Tengah, Indonesia, adalah sebuah monumen megah yang mencerminkan kekayaan warisan budaya daerah tersebut. Dibangun pada abad ke-9 oleh Dinasti Sailendra, bangunan ini tidak hanya berfungsi sebagai tempat ibadah, tetapi juga sebagai keajaiban arsitektur yang mempesona. Berdiri dengan anggun di puncak bukit, candi ini dikelilingi oleh hamparan pepohonan hijau yang menciptakan suasana damai. Kecantikan strukturnya tercermin dalam perpaduan unik antara gaya arsitektur Indonesia dan seni India. Candi ini memiliki sembilan platform bertingkat, yang setiap tingkatnya dihiasi dengan relief batu yang kompleks, menggambarkan ajaran Buddha dan cerita rakyat Jawa. Kompleks ini terdiri dari enam teras berbentuk bujur sangkar dengan tiga pelataran melingkar, yang dihiasi dengan 2.672 panel relief, dan aslinya mencakup 504 arca Buddha. Stupa utama terbesar berada di tengah, memuncakkan struktur ini, dikelilingi oleh tiga barisan melingkar 72 stupa berlubang yang menampung arca Buddha tengah duduk dalam posisi teratai sempurna dengan mudra Dharmachakra (memutar roda dharma). Sejarah mencatat bahwa Borobudur ditinggalkan pada abad ke-10 ketika pusat Kerajaan Mataram Kuno dipindahkan ke Jawa Timur oleh Pu Sindok. Kesadaran global terhadap keberadaan candi ini muncul setelah penemuan pada tahun 1814 oleh Sir Thomas Stamford Raffles, yang saat itu menjabat sebagai Gubernur Jenderal Inggris di Jawa. Sampai saat ini, Borobudur tetap menjadi tempat ziarah keagamaan, di mana setiap tahun umat Buddha dari seluruh Indonesia dan luar negeri berkumpul untuk memperingati Trisuci Waisak.
Candi Borobudur, yang berlokasi di Magelang, Jawa Tengah, Indonesia, merupakan monumen megah yang mencitrakan kekayaan warisan budaya daerah tersebut. Didirikan pada abad ke-9 oleh Dinasti Sailendra, bangunan ini bukan hanya tempat ibadah, tetapi juga karya arsitektur yang menakjubkan. Elegan berdiri di puncak bukit, candi ini dikelilingi oleh hutan hijau yang menciptakan suasana damai. Keindahan strukturnya tercermin dalam perpaduan antara arsitektur Indonesia dan seni India. Terdiri dari sembilan platform bertingkat, setiap tingkatnya dihiasi dengan relief batu kompleks yang menceritakan ajaran Buddha dan cerita rakyat Jawa. Kompleks ini terdiri dari enam teras berbentuk bujur sangkar dengan tiga pelataran melingkar, dihiasi dengan 2.672 panel relief, dan aslinya mengandung 504 arca Buddha. Stupa utama terbesar berada di tengah, memuncakkan bangunan ini, dikelilingi oleh tiga baris melingkar 72 stupa berlubang yang menampung arca Buddha tengah duduk dalam posisi teratai dengan mudra Dharmachakra (memutar roda dharma). Sejarah mencatat bahwa Borobudur ditinggalkan pada abad ke-10 saat pusat Kerajaan Mataram Kuno dipindahkan ke Jawa Timur oleh Pu Sindok. Kesadaran global tentang keberadaan candi ini muncul setelah penemuan pada tahun 1814 oleh Sir Thomas Stamford Raffles, yang saat itu menjabat sebagai Gubernur Jenderal Inggris di Jawa. Hingga kini, Borobudur tetap menjadi tempat ziarah keagamaan, di mana setiap tahun umat Buddha dari seluruh Indonesia dan luar negeri berkumpul untuk merayakan Trisuci Waisak.
Candi Borobudur, yang terletak di Magelang, Jawa Tengah, Indonesia, adalah sebuah monumen bersejarah yang merupakan simbol kekayaan budaya Indonesia. Dibangun pada abad ke-9 oleh Dinasti Sailendra, candi ini awalnya berfungsi sebagai tempat ibadah Buddha, tetapi juga memiliki nilai arsitektur yang luar biasa. Candi Borobudur terletak di puncak bukit yang dikelilingi oleh pepohonan hijau. Perpaduan unik antara arsitektur Indonesia dan seni India tercermin dalam keindahan struktur candi ini. Candi ini memiliki sembilan tingkat yang dihiasi dengan relief batu yang kompleks. Relief-relief ini menggambarkan ajaran Buddha dan cerita rakyat Jawa. Kompleks candi terdiri dari enam teras berbentuk bujur sangkar dengan tiga pelataran melingkar. Dindingnya dihiasi dengan 2.672 panel relief, dan aslinya memiliki 504 arca Buddha. Stupa utama terbesar berada di tengah, dikelilingi oleh tiga barisan melingkar 72 stupa berlubang yang menampung arca Buddha tengah duduk bersila dalam posisi teratai sempurna dengan mudra Dharmachakra (memutar roda dharma). Candi Borobudur ditinggalkan pada abad ke-10 ketika pusat Kerajaan Mataram Kuno dipindahkan ke Jawa Timur. Candi ini ditemukan kembali pada tahun 1814 oleh Sir Thomas Stamford Raffles, yang saat itu menjabat sebagai Gubernur Jenderal Inggris di Jawa. Sejak saat itu, Candi Borobudur menjadi salah satu destinasi wisata paling populer di Indonesia. Borobudur masih menjadi tempat ziarah keagamaan bagi umat Buddha dari seluruh dunia. Setiap tahun, umat Buddha berkumpul di Candi Borobudur untuk memperingati Trisuci Waisak, hari suci umat Buddha yang memperingati kelahiran, pencerahan, dan kematian Buddha Gautama.
Candi Borobudur, yang berlokasi di Magelang, Jawa Tengah, Indonesia, merupakan sebuah monumen spektakuler yang mencerminkan kekayaan warisan budaya daerah tersebut. Dibangun pada abad ke-9 oleh Dinasti Sailendra, bangunan ini bukan hanya sebagai tempat ibadah, melainkan juga sebagai karya arsitektur yang memukau. Terletak dengan anggun di puncak bukit, candi ini dikelilingi oleh hamparan pepohonan hijau yang menciptakan suasana damai. Keelokan strukturnya mencerminkan perpaduan unik antara gaya arsitektur Indonesia dan seni India. Candi ini memiliki sembilan platform bertingkat, di setiap tingkatnya dihiasi dengan relief batu yang kompleks, menggambarkan ajaran Buddha dan kisah rakyat Jawa. Kompleks ini terdiri dari enam teras berbentuk bujur sangkar dengan tiga pelataran melingkar, dihiasi dengan 2.672 panel relief, dan aslinya memuat 504 arca Buddha. Stupa utama terbesar berada di tengah, menjadi puncak bangunan ini, dan dikelilingi oleh tiga baris melingkar 72 stupa berlubang yang menampung arca Buddha tengah duduk dalam posisi teratai sempurna dengan mudra Dharmachakra (memutar roda dharma). Sejarah mencatat bahwa Borobudur ditinggalkan pada abad ke-10 ketika pusat Kerajaan Mataram Kuno dipindahkan ke Jawa Timur oleh Pu Sindok. Kesadaran global terhadap keberadaan candi ini muncul setelah penemuan pada tahun 1814 oleh Sir Thomas Stamford Raffles, yang saat itu menjabat sebagai Gubernur Jenderal Inggris di Jawa. Sampai saat ini, Borobudur masih menjadi tempat ziarah keagamaan, di mana setiap tahun umat Buddha dari seluruh Indonesia dan luar negeri berkumpul untuk merayakan Trisuci Waisak.
"""

qa_pairs=[
    {"question": "Dimana letak Candi Borobudur?", "answer": "Candi Borobudur terletak di Magelang, Jawa Tengah, Indonesia"},
    {"question": "Kapan Candi Borobudur dibangun?", "answer": "Candi Borobudur dibangun pada abad ke-9"},
    {"question": "Siapa yang membangun Candi Borobudur?", "answer": "Candi Borobudur dibangun oleh Dinasti Sailendra"},
    {"question": "Apa fungsi utama Candi Borobudur?", "answer": "Candi Borobudur digunakan sebagai tempat ibadah ajaran Buddha"},
    {"question": "Selain menjadi tempat ibadah ajaran Buddha, apa fungsi Candi Borobudur?", "answer": "Selain sebagai tempat ibadah, Candi Borobudur juga berfungsi sebagai keajaiban arsitektur."},
    {"question": "Bagaimana deskripsi suasana di sekitar Candi Borobudur, khususnya di puncak bukit?", "answer": "Candi ini berdiri elegan di puncak bukit, dikelilingi oleh pepohonan hijau yang menciptakan suasana damai."},
    {"question": "Apa yang membuat Candi Borobudur menjadi sebuah keajaiban arsitektur?", "answer": "Candi Borobudur menjadi keajaiban arsitektur karena perpaduan unik antara arsitektur Indonesia dan seni India."},
    {"question": "Berapa jumlah platform bertingkat yang dimiliki oleh Candi Borobudur?", "answer": "Candi Borobudur memiliki sembilan platform bertingkat."},
    {"question": "Apa yang dihiasi oleh relief batu yang kompleks di setiap tingkat Candi Borobudur?", "answer": "Relief batu yang kompleks menggambarkan ajaran Buddha dan cerita rakyat Jawa."},
    {"question": "Bagaimana struktur kompleks Candi Borobudur terdiri?", "answer": "Struktur kompleks Candi Borobudur terdiri dari enam teras berbentuk bujur sangkar dengan tiga pelataran melingkar."},
    {"question": "Berapa jumlah panel relief yang menghiasi dinding Candi Borobudur?", "answer": "Dinding Candi Borobudur dihiasi dengan 2.672 panel relief."},
    {"question": "Berapa jumlah arca Buddha yang menyertai Candi Borobudur?", "answer": "Candi Borobudur menyertakan 504 arca Buddha."},
    {"question": "Di mana stupa utama terbesar Candi Borobudur berada?", "answer": "Stupa utama terbesar Candi Borobudur berada di tengah, memahkotai bangunan ini."},
    {"question": "Bagaimana stupa utama tersebut dihiasi?", "answer": "Stupa utama dikelilingi oleh tiga barisan melingkar 72 stupa berlubang yang menampung arca Buddha tengah duduk bersila dalam posisi teratai sempurna dengan mudra Dharmachakra (memutar roda dharma)."},
    {"question": "Siapa yang menemukan kembali keberadaan Candi Borobudur pada tahun 1814?", "answer": "Candi Borobudur ditemukan kembali pada tahun 1814 oleh Sir Thomas Stamford Raffles."},
    {"question": "Apa jabatan dari Sir Thomas Stamford Raffles?", "answer": "Gubernur Jenderal Inggris di Jawa"},
    {"question": "Apa posisi dan mudra yang ditunjukkan oleh arca Buddha di dalam 72 stupa berlubang?", "answer": "Arca Buddha dalam 72 stupa berlubang duduk bersila dalam posisi teratai sempurna dengan mudra Dharmachakra (memutar roda dharma)."},
    {"question": "Kapan Candi Borobudur ditinggalkan?", "answer": "Candi Borobudur ditinggalkan pada abad ke-10"},
    {"question": "Mengapa Candi Borobudur ditinggalkan", "answer": "Candi Borobudur ditinggalkan pada abad ke-10 karena perpindahan pusat Kerajaan Mataram Kuno dipindahkan ke Jawa Timur oleh Pu Sindok."},
    {"question": "Apa peran Candi Borobudur dalam kehidupan keagamaan saat ini?", "answer": "Hingga kini, Candi Borobudur masih menjadi tempat ziarah keagamaan, di mana umat Buddha dari seluruh Indonesia dan mancanegara berkumpul untuk memperingati Trisuci Waisak."}
]

import evaluate
def calculate_bleu_score(standard_answer,generated_answer):
  # Load the BLEU evaluation metric
  # kemungkinan default 1, (ngram)
  bleu = evaluate.load("bleu")

  predictions = [generated_answer]
  references = [standard_answer]

  # Compute the BLEU score
  results = bleu.compute(predictions=predictions,references=references)

  return results

def calculate_rouge_score(standard_answer,generated_answer):
  # Load the ROUGE evaluation metric
  rouge = evaluate.load('rouge')

  predictions = [generated_answer]
  references = [standard_answer]

  # Compute the ROUGE score
  results = rouge.compute(predictions=predictions,references=references)

  return results

model_id = GenerationModel.AYA8B.value
chain = get_chain(model_id)

columns=[["question", "expected answer", "generated answer", "blue","rouge", "rougeL", "time exec"]]
measurement = []
for qa in qa_pairs:
  question = qa_pairs["question"]
  ground_truth = qa_pairs["answer"]

  start = time()
  answer = chain.invoke({
    "context": context,
    "messages": qa
  })
  end = time()

  bleu_score = calculate_bleu_score(ground_truth, answer)
  rouge_score = calculate_rouge_score(ground_truth, answer)
  inference_time = end-start
  measurement.append(question, ground_truth, answer, bleu_score["bleu"], rouge_score["rouge2"], rouge_score["rougeL"], [round(inference_time,3)])

import pandas as pd
df = pd.DataFrame(measurement, columns=columns)
print(df.head())
df.to_csv(f"evaluation_text_generation/{model_id}/evaluation.csv")

