from langchain_community.embeddings import HuggingFaceEmbeddings

def load_model(model_path: str):
    model_kwargs = {'trust_remote_code': True, 'device': 'cpu' }
    return HuggingFaceEmbeddings(model_name=model_path, model_kwargs=model_kwargs)
