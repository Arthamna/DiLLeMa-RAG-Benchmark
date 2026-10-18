from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from app.docs import load_pdf
from app.model import load_model

pdf_init_path = "./app/docs"
UUD = load_pdf(f'{pdf_init_path}/UUD-Nomor-Tahun-1945-UUD1945.pdf')
PKM = load_pdf(f'{pdf_init_path}/Panduan-PKM-KI-2024.pdf')
AI = load_pdf(f'{pdf_init_path}/what is AI.pdf')

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

UUD_splits = text_splitter.split_documents(UUD)
PKM_splits = text_splitter.split_documents(PKM)
AI_splits = text_splitter.split_documents(AI)

embedding_firqaa = load_model("./app/model/modules/indo-sentence-bert-base")

UUD_fvs = FAISS.from_documents(UUD_splits, embedding_firqaa)
PKM_fvs = FAISS.from_documents(PKM_splits, embedding_firqaa)
AI_fvs = FAISS.from_documents(AI_splits, embedding_firqaa)

fvs_init_path="./app/embeddings"
UUD_fvs.save_local(folder_path=f"{fvs_init_path}/UUD")
PKM_fvs.save_local(folder_path=f"{fvs_init_path}/PKM")
AI_fvs.save_local(folder_path=f"{fvs_init_path}/AI")