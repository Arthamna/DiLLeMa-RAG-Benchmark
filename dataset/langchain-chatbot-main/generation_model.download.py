from models_generation import GenerationModel, prefix
import os 

from huggingface_hub import snapshot_download, login
from dotenv import load_dotenv

load_dotenv()

login(os.getenv("HUGGINGFACEHUB_API_TOKEN"))

model = GenerationModel.MISTRAL7B.value
snapshot_download(repo_id=model, local_dir=f"{prefix}{model}")
