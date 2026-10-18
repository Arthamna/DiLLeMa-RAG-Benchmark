import os 

from huggingface_hub import snapshot_download, login, HfApi, Repository, create_repo
from dotenv import load_dotenv

from models import models, ModelName

load_dotenv()

login(os.getenv("HUGGINGFACEHUB_API_TOKEN"))

model_directory = "app/model/modules/multilingual-e5-small-finetuning-5"
repo_name = "finetuning-MiniLM-L12-v2"
username = "ramamimu" 
repo_id = f"{username}/{repo_name}"

api = HfApi()
api.upload_folder(
  folder_path=model_directory,
  repo_id=repo_id,
  commit_message="initial commit",
  path_in_repo="."
)