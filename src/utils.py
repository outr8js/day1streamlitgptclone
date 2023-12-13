import gc
from conversation import Conversation_RAG
from vector_index import *
from setup import ModelSetup
import json

def load_models(model_name):
    global conv_qa 
    conv_qa = Conversation_RAG(model_name)
    global model_setup
    model_setup = ModelSetup(model_name)
    success_prompt = model_setup.setup()
    return success_prompt

def get_chat_history(inputs):

    res = []
    for human, ai in inputs:
        res.append(f"Human:{human}\nAssistant:{ai}")
    return "\n".join(res)

def add_text(history, text):

    history = history + [[text, None]]
    return history, ""


def bot(history,
        instruction="Use the following pieces of context to answer the question at the end. Generate the answer based on the given context only if you find the answer in the context. If you do not find any information related to the question in the given context, just say that you don't know, don't try to make up an answer. Keep your answer expressive.",
        temperature=0.1,
        max_new_tokens=512,
        k_context=5,
        ):
    
    instruction = load_prompt('prompts.json', instruction)

    model = conv_qa.create_model(max_new_tokens=max_new_tokens, temperature=temperature)
                             
    qa = conv_qa.create_conversation(
                             model=model,
                             vectordb=model_setup.vectordb,
                             k_context=k_context,
                             instruction=instruction
    )

    chat_history_formatted = get_chat_history(history[:-1])
    res = qa(
        {
            'question': history[-1][0],
            'chat_history': chat_history_formatted
        }
    )

    history[-1][1] = res['answer']
    return history

def clear_cuda_cache():

    gc.collect()
    return None

def load_prompts_list_from_json(json_filepath):
    with open(json_filepath, 'r') as file:
        data = json.load(file)
    return list(data.keys())

def load_prompt(json_filepath, key):
    with open(json_filepath, 'r') as file:
        data = json.load(file)
    return data.get(key, key)