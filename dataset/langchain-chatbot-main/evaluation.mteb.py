from typing import List
from mteb import MTEB
from sentence_transformers import SentenceTransformer
from models import models, ModelName

def mteb_evaluate(model_name:str, tasks: List[str]):
    # model_path = models[model_name]["local_dir"]
    model_path = 'output_path_to_save_model-2e-05'
    model = SentenceTransformer(model_path, trust_remote_code=True)

    # Create MTEB evaluation instance with specific tasks
    evaluation = MTEB(tasks=specific_tasks)

    # Manually print available tasks from MTEB
    print("Available tasks in MTEB:")
    for task in evaluation.tasks:
        print(f"- {task}")


    # Run MTEB benchmark with the specified model and output path
    # evaluation.run(model=model, output_folder=f"./results-mteb/{model_name}")
    evaluation.run(model=model, output_folder=f"./results-mteb/{model_path}")


# https://github.com/embeddings-benchmark/mteb/blob/main/docs/tasks.md
# MULTILINGUAL_E5_SMALL best in task IndonesianIdClickbaitClassification
# multilingual minilm best in task IndonesianMongabayConservationClassification
specific_tasks = ["STSBenchmark"]
# for mdl in models.keys():
mdl=ModelName.MULTILINGUAL_MINILM_FINETUNING_EARLY_STOP.value
print(f"evaluate {mdl}")
mteb_evaluate(mdl, specific_tasks)

