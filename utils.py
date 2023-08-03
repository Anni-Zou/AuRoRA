import re
import random
import torch
import numpy as np
import json

# set the random seed for reproducibility
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data(datatype):
    questions = []
    answers = []
    decoder = json.JSONDecoder()

    if datatype == "gsm8k":
        dataset_path = "./dataset/grade-school-math/test.jsonl"
        with open(dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                questions.append(json_res["question"].strip())
                answers.append(json_res["answer"].split("#### ")[-1].replace(",", ""))
    elif datatype == "aqua":
        dataset_path = "./dataset/AQuA/test.json"
        with open(dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                qes = json_res["question"].strip() + " Answer Choices:"

                for opt in json_res["options"]:
                    opt = opt.replace(')', ') ')
                    qes += f" ({opt}"

                questions.append(qes)
                answers.append(json_res["correct"])
    elif datatype == "svamp":
        dataset_path = "./dataset/SVAMP/SVAMP.json"
        with open(dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["Body"].strip() + " " + line["Question"].strip()
                a = str(line["Answer"])
                if a[-2:] == ".0":
                    a = a[:-2]
                questions.append(q)
                answers.append(a)
    elif datatype in ("addsub", "singleeq", "multiarith"):
        if datatype == "addsub":
            dataset_path = "./dataset/AddSub/AddSub.json"
        elif datatype == "singleeq":
            dataset_path = "./dataset/SingleEq/questions.json"
        elif datatype == "multiarith":
            dataset_path = "./dataset/MultiArith/MultiArith.json"
        with open(dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["sQuestion"].strip()
                a = str(line["lSolutions"][0])
                if a[-2:] == ".0":
                    a = a[:-2]
                questions.append(q)
                answers.append(a)
    elif datatype == "commonsensqa":
        dataset_path = "./dataset/CommonsenseQA/dev_rand_split.jsonl"
        with open(dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                choice = "Answer Choices:"
                for c in json_res["question"]["choices"]:
                    choice += " ("
                    choice += c["label"]
                    choice += ") "
                    choice += c["text"]
                questions.append(json_res["question"]["stem"].strip() + " " + choice)
                answers.append(json_res["answerKey"])
    elif datatype == "csqa2":
        dataset_path = "./dataset/CSQA2/CSQA2_test_no_answers.json"
        with open(dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = json.loads(f)
                q = json_res["question"]
                a = ""
                questions.append(q)
                answers.append(a)

    elif datatype == "strategyqa":
        dataset_path = "./dataset/StrategyQA/task.json"
        if 'task' in dataset_path:
            with open(dataset_path) as f:
                json_data = json.load(f)["examples"]
                for line in json_data:
                    q = line["input"].strip()
                    a = int(line["target_scores"]["Yes"])
                    if a == 1:
                        a = "yes"
                    else:
                        a = "no"
                    questions.append(q)
                    answers.append(a)
        else:
            with open(dataset_path, encoding='utf-8') as f:
                json_data = json.load(f)
                for line in json_data:
                    q = line["question"].strip() 
                    if line['answer']:
                        a = 'yes'
                    else:
                        a = 'no'
                    questions.append(q)
                    answers.append(a)
    elif datatype in ("coin_flip", "last_letters"):
        if datatype == "coin_flip":
            dataset_path = "./dataset/coin_flip/coin_flip.json"
        elif datatype == "last_letters":
            dataset_path = "./dataset/last_letters/last_letters.json"
        with open(dataset_path) as f:
            json_data = json.load(f)
            json_data = json_data["examples"]
            for line in json_data:
                q = line["question"]
                a = line["answer"]
                questions.append(q)
                answers.append(a)
    else:
        raise NotImplementedError

    print(f"dataset: {datatype}")
    print(f"dataset_size: {len(answers)}")
    
    return questions, answers


# return a customized dataloader of batches
# Not PyTorch dataloader, it supprts random index(slice) access
def create_dataloader(datatype)->list:
    set_random_seed(42)
    questions, answers = load_data(datatype)
    dataset = []
    for idx in range(len(questions)):
        dataset.append({"question":questions[idx], "answer":answers[idx], "question_idx":idx})

    random.shuffle(dataset)
    print(f"dataloader size: {len(dataset)}")

    return dataset



def answer_cleansing_zero_shot(dataset, pred, must_choice=False):
    pred = pred.strip()
    if dataset in ("aqua", "commonsensqa"):
        pred = re.findall(r'A|B|C|D|E', pred)
    elif dataset == "bigbench_date":
        pred = re.findall(r'A|B|C|D|E|F', pred)
    elif dataset in ("object_tracking"):
        pred = re.findall(r'A|B|C', pred)
    elif dataset in ("gsm8k", "addsub", "multiarith", "svamp", "singleeq"):
        if must_choice:
            pred = re.findall(r'A|B|C|D', pred)
        else:
            pred = pred.replace(",", "")
            pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]
    elif dataset in ("strategyqa", "coin_flip"):
        pred = pred.lower()
        pred = re.sub("\"|\'|\n|\.|\s|\:|\,", " ", pred)
        pred = pred.split(" ")
        pred = [i for i in pred if i in ("yes", "no")]
    elif dataset == "last_letters":
        pred = re.sub("\"|\'|\n|\.|\s", "", pred)
        pred = [pred]
    else:
        raise ValueError("dataset is not properly defined ...")

    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = ""
    else:
        # choose the first element in list ...
        pred = pred[0]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]

    return pred

def entity_cleansing(ent):
    ent = re.sub("\n|\s*-\s*|\.", ",", ent)
    ent = ent.split(",")
    ent_ = []
    for e in ent:
        e = e.strip()
        if e != "" and e not in ('A', 'B', 'C', 'D', 'E'):
            ent_.append(e)
    return ent_

def knowledge_cleansing(knowledge):
    #print("Knowledge Before: " + knowledge)
    knowledge = knowledge.strip()
    if knowledge.startswith("No, "):
        knowledge = re.sub("No, ", "", knowledge)
    knowledge = re.sub("\s"," ", knowledge)
    #print("Knowledge After: " + knowledge)
    return knowledge