from langchain_community.document_loaders import PyPDFLoader

def load_pdf(pdf_path: str):
  return PyPDFLoader(pdf_path).load()