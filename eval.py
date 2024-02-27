from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from torch.utils.data import random_split
import torch
import datasets
from peft import PeftConfig, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
import numpy as np
import datasets
from transformers.integrations import WandbCallback
import warnings
import random 

import os 
model_name_t = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
model_name = 'emb_real/checkpoint-100/'
tokenizer = AutoTokenizer.from_pretrained(model_name_t, padding_side='right')
        # add doc_emb token
tokenizer.doc_emb_token = '[DOC_EMB]'
tokenizer.add_tokens([tokenizer.doc_emb_token], special_tokens=False) 
tokenizer.doc_emb_id = tokenizer.encode(tokenizer.doc_emb_token, add_special_tokens=False)[0]
tokenizer.pad_token = tokenizer.bos_token

dataset = datasets.load_from_disk('/local/calmar/rag/datasets/kilt-100w_full/').shuffle(seed=41).select(range(100))



quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_compute_dtype='bfloat16',
                )
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    quantization_config=quant_config, 
    attn_implementation="flash_attention_2",
    device_map='auto',
)

# model.merge_and_unload()
model.config.use_cache = False
model = model.bfloat16()
model.config.pretraining_tp = 1
model.resize_token_embeddings(len(tokenizer))
doc_max_length = 16
def collate_fn(examples):
    inp = [tokenizer.doc_emb_token +' '.join(e['content'].split(' ')[:doc_max_length])+ tokenizer.eos_token for e in examples]
    inp = tokenizer(inp, return_tensors='pt', padding=True)
    label = inp['input_ids']
    inp.update({'label': label})
    return inp

os.environ["WANDB_PROJECT"] = 'dec_from_emb'

def get_eos_repr(model, inp):
    emb = model(**inp.to('cuda'), output_hidden_states=True).hidden_states[-1]
    mask = inp['input_ids'] == tokenizer.eos_token_id
    return emb[mask].unsqueeze(1)

def replace_embeddings(input_ids, inputs_embeds, insert_embeds, placeholder_id):
    insert_embeds = insert_embeds.to(inputs_embeds.device)
    # get indexes of placeholder ids
    replace_index = (input_ids == placeholder_id).nonzero(as_tuple=True)[1][0]
    inputs_embeds = torch.cat((inputs_embeds[:, :replace_index], insert_embeds, inputs_embeds[:, replace_index+1:] ), 1)
    return inputs_embeds

    
from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
for inputs in dataloader:
    input_ids = inputs['input_ids'].to('cuda')
    attention_mask = inputs['input_ids'].to('cuda')
    label = inputs.pop('label')
    eos_emb = get_eos_repr(model, inputs)
    inputs_embeds = model.get_input_embeddings()(torch.LongTensor([tokenizer.bos_token_id] * eos_emb.size(0)).unsqueeze(1).to('cuda'))
    inputs_embeds = torch.cat((inputs_embeds, eos_emb), 1)
    out = model.generate(inputs_embeds=inputs_embeds, do_sample=False, max_new_tokens=128)
    print('label', tokenizer.batch_decode(label))
    print('gen', tokenizer.batch_decode(out))
    print()
    

