import openai # For GPT-3 API ...
import json
from collections import Counter
from llm_utils import *
from utils import *
from retrieval_utils import *

openai.api_key = ""     #input your own api key here

COT_PROMPT = "Let's think step by step."
DIRECT_ANS_PROMPT = "The answer is"

def answer_extraction_prompt(datatype):
    if datatype == "commonsensqa":
        ans_prompt = "\nTherefore, among A through E, the answer is"
    elif datatype == "strategyqa":
        ans_prompt = "\nTherefore, the answer (Yes or No) is"
    elif datatype == "multiarith" or datatype == "singleeq":
        ans_prompt = "\nTherefore, the answer (arabic numerals) is"
    elif datatype == "coin_flip":
        ans_prompt = "\nTherefore, the answer (Yes or No) is"
    elif datatype == "last_letters":
        ans_prompt = "\nTherefore, the answer is"
    return ans_prompt


def zero_shot(datatype, question, engine):
    ANS_EXTRACTION_PROMPT = answer_extraction_prompt(datatype)
    ANS_EXTRACTION_PROMPT = ANS_EXTRACTION_PROMPT.replace("\nTherefore, ", "")
    ANS_EXTRACTION_PROMPT = ANS_EXTRACTION_PROMPT[0].upper() + ANS_EXTRACTION_PROMPT[1:]

    input = "Q: " + question + " " + "\n" + "A: " + ANS_EXTRACTION_PROMPT
    response = decoder_for_gpt3(input, max_length=32, engine=engine)
    response = answer_cleansing_zero_shot(datatype, response)
    return response

def read_demo(datatype):
    if datatype == "multiarith":
        demo_path = './demos/multiarith'
    elif datatype == "commonsensqa":
        demo_path = './demos/commonsensqa'
    elif datatype == "strategyqa":
        demo_path = './demos/strategyqa'
    elif datatype == "coin_flip":
        demo_path = './demos/coin_flip'
    elif datatype == "last_letters":
        demo_path = './demos/last_letters'
    elif datatype == "singleeq":
        demo_path = './demos/singleeq'
    else:
        pass

    ##读取对应的demo
    x, z, y =[], [], []
    with open(demo_path, encoding="utf-8") as f:
        json_data = json.load(f)
        json_data = json_data["demo"]
        for line in json_data:
            x.append(line["question"])
            z.append(line["rationale"])
            y.append(line["pred_ans"])
    index_list = list(range(len(x)))

    demo_text = ""
    for i in index_list:
        demo_text += x[i] + " " + z[i] + " " + \
                    DIRECT_ANS_PROMPT + " " + y[i] + ".\n\n"

    return demo_text
        
def auto_cot_consi(question, demo_text, engine):
    input = demo_text + "Q: " + question + "\n" + "A: " + COT_PROMPT
    cot_responses = decoder_for_gpt3_consistency(input,max_length=256, engine=engine) #list of cots
    return cot_responses


def zero_shot_cot(datatype, question):
    ANS_EXTRACTION_PROMPT = answer_extraction_prompt(datatype)
    input = "Q: " + question + "\n" + "A: " + COT_PROMPT
    cot_response = decoder_for_gpt3(input, max_length=256)
    input = input + cot_response + ANS_EXTRACTION_PROMPT
    ans_response = decoder_for_gpt3(input, max_length=32)
    ans_response = answer_cleansing_zero_shot(datatype, ans_response)
    #if ans_response == "":
    #    ans_response = "VOID"
    return cot_response, ans_response


def zero_cot_consi(question):
    input = "Q: " + question + "\n" + "A: " + COT_PROMPT
    cot_responses = decoder_for_gpt3_consistency(input,max_length=256) #list of cots
    return cot_responses

def cot_revision(datatype, question, ori_cots, knowledge, engine):
    ANS_EXTRACTION_PROMPT = answer_extraction_prompt(datatype)
    corrected_rationales = []
    corrected_answers = []
    correction_prompt = "Question: " + "[ " + question + "]\n"
    correction_prompt += "Knowledge: " + "[ " + knowledge + "]\n"
    for ori_r in ori_cots:
        cor_p = correction_prompt + "Original rationale: " + "[ " + ori_r + "]\n"
        cor_p += "With Knowledge given, output the revised rationale for Question in a precise and certain style by thinking step by step: "
        corrected_rationale = decoder_for_gpt3(cor_p,max_length=256, temperature=0.7, engine=engine)
        corrected_rationale = corrected_rationale.strip()
        corrected_rationales.append(corrected_rationale)
        input = "Q: " + question + "\n" + "A: " + corrected_rationale + ANS_EXTRACTION_PROMPT
        ans = decoder_for_gpt3(input, max_length=32, temperature=0.7, engine=engine)
        ans = answer_cleansing_zero_shot(datatype, ans)
        corrected_answers.append(ans)
    return corrected_rationales, corrected_answers


def consistency(arr):
    len_ans = len(arr)
    arr_acounts = Counter(arr)
    ans_freq_tuple = arr_acounts.most_common(len_ans)
    most_frequent_item, _ = ans_freq_tuple[0]
    ans_dict = {}
    for ans_freq in ans_freq_tuple:
        ans, times = ans_freq
        ans_dict[ans] = times/len_ans
    return most_frequent_item, ans_dict



def main(datatype, engine, resume_id, limit_size=500):
    #datatype, input_question
    dataset = create_dataloader(datatype)

    output_path = f"./experiment/{datatype}_5paths"
    with open(output_path, 'a') as f:
        for i, data in enumerate(dataset):
            if i < resume_id -1:
                continue
            print('*************************')
            print("{}st data".format(i+1))

            input_question = data['question']
            gold_ans = data['answer']

            #self-construction
            demo_text = read_demo(datatype)
            #self-retrieval
            entities, self_retrieve_knowledge, kb_retrieve_knowledge = retrieve_for_question(input_question, engine)
            #self-refinement
            refined_knowledge = refine_for_question(input_question, self_retrieve_knowledge, kb_retrieve_knowledge, engine)
            #self-revision
            ori_cots = auto_cot_consi(input_question, demo_text, engine)
            cor_cots, cor_ans = cot_revision(datatype, input_question, ori_cots, refined_knowledge, engine)
            #self-consistency
            our_ans, ans_dict = consistency(cor_ans)
            zeroshot_ans = zero_shot(datatype, input_question, engine)

            single_data = {
                'question': input_question,
                'gold_ans': gold_ans,
                'dataset_idx': data['question_idx'],
                'pred_ans': our_ans,
                'zero_ans': zeroshot_ans,
                'ori_cots': ori_cots,
                'cor_cots': cor_cots,
                'refined_know': refined_knowledge,
                'self_know': self_retrieve_knowledge,
                'kb_know': kb_retrieve_knowledge,
                'ans_dict': ans_dict
            }

            print("------Input_Question------")
            print(input_question)
            print("------Gold_Ans------")
            print(gold_ans)
            print("------Adapter_Ans------")
            print(our_ans)
            print("------ZeroShot_Ans------")
            print(zeroshot_ans)



            record = json.dumps(single_data)
            f.write(record + '\n')
            
            if (limit_size != 0) and ((i+1)>limit_size):
                break


    return 

if __name__ == "__main__":
    limit_size = 0
    resume_id = 0
    engine = "text-davinci-003"
    datatype = "multiarith"

    main(datatype, engine, resume_id, limit_size=limit_size)





