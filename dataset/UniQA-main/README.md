# UniQA

UniQA: an Italian and English Question-Answering Data Set Based on Educational Documents

Please cite our work as follows if you use UniQA

```
@inproceedings{siragusa2024uniqa,
	author = {Siragusa, Irene and Pirrone, Roberto},
	title={UniQA: an Italian and English Question-Answering Data Set Based on Educational Documents},
	booktitle={Proceedings of the Eighth Workshop on Natural Language for Artificial Intelligence (NL4AI 2024) co-located with 23th International Conference of the Italian Association for Artificial Intelligence (AI* IA 2024)},
	year = {2024},
	address= {Bolzano, Italy}
}
```

Below a description of the repository:

## Documents

  A total of 1048 documents, both in English and Italian, were collected from the website of the University of Palermo.
  Each document is present in both languages and no translation procedure was performed to obtain the corpus. 

  **unipa-corpus-24-en**
  
  Here are collected all the 524 documents in English, divided in *course-info* and *course-outline* folders, each folders collect 262 documents.

  **unipa-corpus-24-it**
  
  Here are collected all the 524 documents in Italian, divided in *course-info* and *course-outline* folders, each folders collect 262 documents.

## QA pairs

  In *QA-dataset* are collected the created question-answer pairs, and are arranged in json files, structured as a list of dictionary reporting `question`, `answer` and a list of the file names from which the answer is obtained in `context` field, as below:

    {
        "question": "What are the available curriculum for the master degree in computer engineering?",
        "context": [
            "2035_CYBERSECURITY_dettagli_en.txt",
            "2035_ARTIFICIAL INTELLIGENCE_dettagli_en.txt"
        ],
        "answer": "For the master degree in computer engineering are available the following curriculum: cybersecurity, artificial intelligence. For more information visit the course website http://www.unipa.it/struttura.html?id=721."
    }

  Each pair is present in both languages and was manually created without any automatic translation procedure. 

  - `qa_info_en.json` and `qa_info_it.json` contains all QA pairs related to course-info documents, 1520 per document, in English and Italian, respectively.
  - `qa_outline_en.json` and `qa_outline_it.json` contains all QA pairs related to course-outline documents, 5351 per document, in English and Italian, respectively.
  - `qa_splits_en.json` and `qa_splits_it.json` contains a train-test split for English and Italian, respetively. File format is suitable for loading the dataset as a Dataset object in datasets library from huggingface
 - `qa_splits_joint.json` contains a train-test split for both English and Italian. File format is suitable for loading the dataset as a Dataset object in datasets library from huggingface
 - `train_test_split_codes.json` and `train_test_split_files.json` contain the courses' codes and files names, used to create the train-test split.

**Note** train-test split is the same for both languages, it was created from a 80-20 split assuring a equal division along different departments. 
