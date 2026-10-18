import pandas as pd

from langchain_community.vectorstores import FAISS
from app.model import load_model
from models import ModelName, models
from langchain.embeddings.openai import OpenAIEmbeddings

model_names = [
  ModelName.MULTILINGUAL_MINILM_FINETUNING_192_b8.value
]

for model_name in model_names:
  embedding_model = OpenAIEmbeddings() if model_name == ModelName.GPT3_TURBO.value else load_model(models[model_name]["local_dir"])

  from langchain.text_splitter import RecursiveCharacterTextSplitter
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size = 1000,
      chunk_overlap = 200
  )

  def embed_content(content, title, doc_type):
    splitted_texts = text_splitter.split_text(content)

    metadata = {"source": title, "type": doc_type}
    metadatas = [metadata] * len(splitted_texts)
    if len(metadatas) <= 0:
      return
    embedded_texts = FAISS.from_texts(splitted_texts, embedding_model, metadatas=metadatas)

    fvs_init_path=f"./app/embeddings/{model_name}"
    embedded_texts.save_local(folder_path=f"{fvs_init_path}/{title}")

    print(f"successfully embed and save {title} in {fvs_init_path}")


  counter = 0
  def start_embed(df):
    global counter
    counter+=1
    print(counter)
    for _, row in df.iterrows():
      title = row['Judul']
      document_type = row['Jenis Dokumen']
      content_1 = row['Konten 1']
      content_2 = row['Konten 2']

      embed_content(content_1, title, document_type)
      # if second content not empty
      if not pd.isna(row['Konten 2']):
        embed_content(content_2, title, document_type)  


  df_akademik = pd.read_csv('dataset/akademik/Dataset IF-Tegration - akademik.csv')
  df_akademik_s1 = pd.read_csv('dataset/akademik s1/Dataset IF-Tegration - Akademik S1.csv')
  df_akademik_s2 = pd.read_csv('dataset/akademik s2/Dataset IF-Tegration - Akademik S2.csv')
  df_beasiswa = pd.read_csv('dataset/beasiswa/Dataset IF-Tegration - beasiswa.csv')
  df_dana_pendidikan = pd.read_csv('dataset/dana pendidikan/Dataset IF-Tegration - dana pendidikan.csv')
  df_kerja_praktik = pd.read_csv('dataset/kerja praktik/Dataset IF-Tegration - kerja praktik.csv')
  df_magang = pd.read_csv('dataset/magang/Dataset IF-Tegration - magang.csv')
  df_MBKM = pd.read_csv('dataset/MBKM/Dataset IF-Tegration - MBKM.csv')
  df_program_internasional = pd.read_csv('dataset/program internasional/Dataset IF-Tegration - program internasional.csv')
  df_SKEM = pd.read_csv('dataset/SKEM/Dataset IF-Tegration - SKEM.csv')
  df_tesis = pd.read_csv('dataset/tesis/Dataset IF-Tegration - tesis.csv')
  df_wisuda = pd.read_csv('dataset/wisuda/Dataset IF-Tegration - wisuda.csv')
  df_yudisium_dan_tugas_akhir = pd.read_csv('dataset/yudisium dan tugas akhir/Dataset IF-Tegration - yudisium dan tugas akhir.csv')
  df_akademik_luar_kampus = pd.read_csv('dataset/akademik luar kampus/Dataset IF-Tegration - akademik luar kampus.csv')
  df_yudisium = pd.read_csv('dataset/yudisium/Dataset IF-Tegration - yudisium.csv')
  df_dana_perkuliahan = pd.read_csv('dataset/dana perkuliahan/Dataset IF-Tegration - dana perkuliahan.csv')
  df_international_exposure_en = pd.read_csv('dataset/international exposure en/Dataset IF-Tegration - international expo.csv')
  df_international_exposure = pd.read_csv('dataset/international exposure id/Dataset IF-Tegration - international expo.csv')

  start_embed(df_akademik)
  start_embed(df_akademik_luar_kampus)
  start_embed(df_yudisium)
  start_embed(df_dana_perkuliahan)
  start_embed(df_program_internasional)
  start_embed(df_akademik_s1)
  start_embed(df_akademik_s2)
  start_embed(df_international_exposure)
  start_embed(df_international_exposure_en)
  # start_embed(df_beasiswa)
  # start_embed(df_dana_pendidikan)
  # start_embed(df_kerja_praktik)
  # start_embed(df_magang)
  # start_embed(df_MBKM)
  # start_embed(df_SKEM)
  # start_embed(df_tesis)
  # start_embed(df_wisuda)
  # start_embed(df_yudisium_dan_tugas_akhir)
