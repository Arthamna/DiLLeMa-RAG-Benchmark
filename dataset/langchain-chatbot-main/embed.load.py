from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from app.docs import load_pdf
from app.model import load_model

UUD = load_pdf('./app/docs/UUD-Nomor-Tahun-1945-UUD1945.pdf')

embedding_firqaa = load_model("./app/model/modules/indo-sentence-bert-base")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

UUD_splits = text_splitter.split_documents(UUD)
UUD_fvs = FAISS.from_documents(UUD_splits, embedding_firqaa)

question = "Dimakah letak pasal yang berbunyi 'Majelis Permusyawaratan Rakyat terdiri atas anggota-anggota Dewan Perwakilan Rakyat, ditambah dengan utusan-utusan dari daerah-daerah dan golongan-golongan, menurut aturan yang ditetapkan dengan undang-undang.'"
ss = UUD_fvs.similarity_search(question, k=3)

print(ss)