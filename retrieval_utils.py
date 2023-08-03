'''
Modified from https://github.com/RuochenZhao/Verify-and-Edit
'''

import wikipedia
import wikipediaapi
import spacy
import numpy as np
import ngram
#import nltk
import torch
import sklearn
#from textblob import TextBlob
from nltk import tokenize
from sentence_transformers import SentenceTransformer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer
from llm_utils import decoder_for_gpt3
from utils import entity_cleansing, knowledge_cleansing

wiki_wiki = wikipediaapi.Wikipedia('en')
nlp = spacy.load("en_core_web_sm")
ENT_TYPE = ['EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'NORP', 'ORG', 'PERSON', 'PRODUCT', 'WORK_OF_ART']

CTX_ENCODER = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
CTX_TOKENIZER = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base", model_max_length = 512)
Q_ENCODER = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
Q_TOKENIZER = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base", model_max_length = 512)


## todo: extract entities from ConceptNet
def find_ents(text, engine):
    doc = nlp(text)
    valid_ents = []
    for ent in doc.ents:
        if ent.label_ in ENT_TYPE:
            valid_ents.append(ent.text)
    #in case entity list is empty: resort to LLM to extract entity
    if valid_ents == []:
        input = "Question: " + "[ " + text + "]\n"
        input += "Output the entities in Question separated by comma: "
        response = decoder_for_gpt3(input, 32, engine=engine)
        valid_ents = entity_cleansing(response)
    return valid_ents


def relevant_pages_for_ents(valid_ents, topk = 5):
    '''
    Input: a list of valid entities
    Output: a list of list containing topk pages for each entity
    '''
    if valid_ents == []:
        return []
    titles = []
    for ve in valid_ents:
        title = wikipedia.search(ve)[:topk]
        titles.append(title)
    #titles = list(dict.fromkeys(titles))
    return titles


def relevant_pages_for_text(text, topk = 5):
    return wikipedia.search(text)[:topk]


def get_wiki_objs(pages):
    '''
    Input: a list of list
    Output: a list of list
    '''
    if pages == []:
        return []
    obj_pages = []
    for titles_for_ve in pages:
        pages_for_ve = [wiki_wiki.page(title) for title in titles_for_ve]
        obj_pages.append(pages_for_ve)
    return obj_pages


def get_linked_pages(wiki_pages, topk = 5):
    linked_ents = []
    for wp in wiki_pages:
        linked_ents += list(wp.links.values())
        if topk != -1:
            linked_ents = linked_ents[:topk]
    return linked_ents


def get_texts_to_pages(pages, topk = 2):
    '''
    Input: list of list of pages
    Output: list of list of texts
    '''
    total_texts = []
    for ve_pages in pages:
        ve_texts = []
        for p in ve_pages:
            text = p.text
            text = tokenize.sent_tokenize(text)[:topk]
            text = ' '.join(text)
            ve_texts.append(text)
        total_texts.append(ve_texts)
    return total_texts



def DPR_embeddings(q_encoder, q_tokenizer, question):
    question_embedding = q_tokenizer(question, return_tensors="pt",max_length=5, truncation=True)
    with torch.no_grad():
        try:
            question_embedding = q_encoder(**question_embedding)[0][0]
        except:
            print(question)
            print(question_embedding['input_ids'].size())
            raise Exception('end')
    question_embedding = question_embedding.numpy()
    return question_embedding

def model_embeddings(sentence, model):
    embedding = model.encode([sentence])
    return embedding[0] #should return an array of shape 384

##todo: plus overlap filtering
def filtering_retrieved_texts(question, ent_texts, retr_method="wikipedia_dpr", topk=1):
    filtered_texts = []
    for texts in ent_texts:
        if texts != []: #not empty list
            if retr_method == "ngram":
                pars = np.array([ngram.NGram.compare(question, sent, N=1) for sent in texts])
                #argsort: smallest to biggest
                pars = pars.argsort()[::-1][:topk]
            else:
                if retr_method == "wikipedia_dpr":
                    sen_embeds = [DPR_embeddings(Q_ENCODER, Q_TOKENIZER, question)]
                    par_embeds = [DPR_embeddings(CTX_ENCODER, CTX_TOKENIZER, s) for s in texts]
                else:
                    embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
                    sen_embeds = [model_embeddings(question, embedding_model)]
                    par_embeds = [model_embeddings(s, embedding_model) for s in texts]
                pars = sklearn.metrics.pairwise.pairwise_distances(sen_embeds, par_embeds)
                pars = pars.argsort(axis=1)[0][:topk]

            filtered_texts += [texts[i] for i in pars]
            filtered_texts = list(dict.fromkeys(filtered_texts))
    return filtered_texts

def join_knowledge(filtered_texts):
    if filtered_texts == []:
        return ""
    return " ".join(filtered_texts)

def retrieve_for_question_kb(question, engine, know_type="entity_know", no_links=False):
    valid_ents = find_ents(question, engine)
    print(valid_ents)
    if valid_ents == []:
        return [], ""

    # find pages
    page_titles = []
    if "entity" in know_type:
        pages_for_ents = relevant_pages_for_ents(valid_ents, topk = 5)  #list of list
        if pages_for_ents != []:
            page_titles += pages_for_ents
    if "question" in know_type:
        pages_for_question = relevant_pages_for_text(question, topk = 5)
        if pages_for_question != []:
            page_titles += pages_for_question
    pages = get_wiki_objs(page_titles)  #list of list
    if pages == []:
        return ""
    new_pages = []
    assert page_titles != []
    assert pages != []

    print(page_titles)
    #print(pages)
    for i, ve_pt in enumerate(page_titles):
        new_ve_pages = []
        for j, pt in enumerate(ve_pt):
            if 'disambiguation' in pt:
                new_ve_pages += get_linked_pages([pages[i][j]], topk=-1)
            else:
                new_ve_pages += [pages[i][j]]
        new_pages.append(new_ve_pages)
    
    pages = new_pages
    
    if not no_links:
        # add linked pages
        for ve_pages in pages:
            ve_pages += get_linked_pages(ve_pages, topk=5)
            ve_pages = list(dict.fromkeys(ve_pages))
    #get texts
    texts = get_texts_to_pages(pages, topk=1)
    filtered_texts = filtering_retrieved_texts(question, texts)
    joint_knowledge = join_knowledge(filtered_texts)


    return valid_ents, joint_knowledge

def retrieve_for_question(question, engine, retrieve_source="llm_kb"):
    # Retrieve knowledge from LLM
    if "llm" in retrieve_source:
        self_retrieve_prompt = "Question: " + "[ " + question + "]\n"
        self_retrieve_prompt += "Necessary knowledge about the question by not answering the question: "
        self_retrieve_knowledge = decoder_for_gpt3(self_retrieve_prompt, 256, engine=engine)
        self_retrieve_knowledge = knowledge_cleansing(self_retrieve_knowledge)
        print("------Self_Know------")
        print(self_retrieve_knowledge)
    
    # Retrieve knowledge from KB
    if "kb" in retrieve_source:
        entities, kb_retrieve_knowledge = retrieve_for_question_kb(question, engine, no_links=True)
        if kb_retrieve_knowledge != "":
            print("------KB_Know------")
            print(kb_retrieve_knowledge)
    
    return entities, self_retrieve_knowledge, kb_retrieve_knowledge

def refine_for_question(question, self_retrieve_knowledge, kb_retrieve_knowledge, engine, retrieve_source="llm_kb"):

    # Refine knowledge
    if retrieve_source == "llm_only":
        refine_knowledge = self_retrieve_knowledge
    elif retrieve_source == "kb_only":
        if kb_retrieve_knowledge != "":
            refine_prompt = "Question: " + "[ " + question + "]\n"
            refine_prompt += "Knowledge: " + "[ " + kb_retrieve_knowledge + "]\n"
            refine_prompt += "Based on Knowledge, output the brief and refined knowledge necessary for Question by not giving the answer: "
            refine_knowledge = decoder_for_gpt3(refine_prompt, 256, engine=engine)
            print("------Refined_Know------")
            print(refine_knowledge)
        else:
            refine_knowledge = ""
    elif retrieve_source == "llm_kb":
        if kb_retrieve_knowledge != "":
            #refine_prompt = "Question: " + "[ " + question + "]\n"
            refine_prompt = "Knowledge_1: " + "[ " + self_retrieve_knowledge + "]\n"
            refine_prompt += "Knowledge_2: " + "[ " + kb_retrieve_knowledge + "]\n"
            #refine_prompt += "By using Knowledge_2 to check Knowledge_1, output the brief and correct knowledge necessary for Question: "
            refine_prompt += "By using Knowledge_2 to check Knowledge_1, output the brief and correct knowledge: "
            refine_knowledge = decoder_for_gpt3(refine_prompt, 256, engine=engine)
            refine_knowledge = knowledge_cleansing(refine_knowledge)
            #refine_knowledge = kb_retrieve_knowledge + refine_knowledge
            print("------Refined_Know------")
            print(refine_knowledge)
        else:
            refine_knowledge = self_retrieve_knowledge
    
    return refine_knowledge
