from langchain_community.vectorstores import FAISS

from app.model import load_model

embedding_firqaa = load_model("./app/model/modules/indo-sentence-bert-base")

def get_embeddings(embed_path: str):
  # at least has one embed data
  faiss = FAISS.load_local(folder_path=f"{embed_path}/AI", embeddings=embedding_firqaa, allow_dangerous_deserialization=True)
  PKM_fvs = FAISS.load_local(folder_path=f"{embed_path}/PKM", embeddings=embedding_firqaa, allow_dangerous_deserialization=True)
  UUD_fvs = FAISS.load_local(folder_path=f"{embed_path}/UUD", embeddings=embedding_firqaa, allow_dangerous_deserialization=True)
  faiss.merge_from(PKM_fvs)
  faiss.merge_from(UUD_fvs)
  return faiss