import os 

from huggingface_hub import snapshot_download, login
from dotenv import load_dotenv

from models import models, ModelName

load_dotenv()

login(os.getenv("HUGGINGFACEHUB_API_TOKEN"))


snapshot_download(repo_id=models[ModelName.MULTILINGUAL_E5_SMALL.value]["repo_id"],
                   local_dir=models[ModelName.MULTILINGUAL_E5_SMALL.value]["local_dir"])
