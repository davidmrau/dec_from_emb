from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import datasets
from torch.utils.data import DataLoader
import datasets
import os 
from poc_decode_emb import get_emb_repr, collate_fn
from torch.utils.data import DataLoader
from modeling_llama import LlamaForCausalLM
model_name_t = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
model_name = 'emb_TinyLlama_TinyLlama-1.1B-Chat-v1.0/checkpoint-100/'

tokenizer = AutoTokenizer.from_pretrained(model_name_t, padding_side='right')
 # add tokens
tokenizer.emb_token = '<emb>'
tokenizer.add_tokens([tokenizer.emb_token], special_tokens=False) 
tokenizer.emb_token_id = tokenizer.encode(tokenizer.emb_token, add_special_tokens=False)[0]

tokenizer.agg_token = '<agg>'
tokenizer.add_tokens([tokenizer.agg_token], special_tokens=True) 
tokenizer.agg_token_id = tokenizer.encode(tokenizer.agg_token, add_special_tokens=False)[0]

tokenizer.pad_token = tokenizer.bos_token
dataset = datasets.load_from_disk('/local/calmar/rag/datasets/kilt-100w_full/').shuffle(seed=41).select(range(100))


quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_compute_dtype='bfloat16',
                )

                
# model = AutoModelForCausalLM.from_pretrained(
#     model_name, 
#     quantization_config=quant_config, 
#     attn_implementation="flash_attention_2",
#     device_map='auto',
# )

model = LlamaForCausalLM.from_pretrained(
    model_name, 
    quantization_config=quant_config, 
    device_map='auto',
)

# model.merge_and_unload()
model.config.use_cache = False
model = model.bfloat16()
model.config.pretraining_tp = 1
model.resize_token_embeddings(len(tokenizer))

doc_max_length = 32

os.environ["WANDB_PROJECT"] = 'dec_from_emb'
    

dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
for inputs in dataloader:
    inputs = inputs.to('cuda') 
    input_ids = inputs['input_ids']
    attention_mask = inputs['input_ids']
    label = inputs.pop('label')
    emb = get_emb_repr(model, inputs)
    inputs_embeds = model.get_input_embeddings()(torch.LongTensor([tokenizer.bos_token_id] * emb.size(0)).unsqueeze(1).to('cuda'))
    inputs_embeds = torch.cat((inputs_embeds, emb), 1)
    out = model.generate(inputs_embeds=inputs_embeds, do_sample=False, max_new_tokens=128)
    label[label==-100] = tokenizer.eos_token_id
    print('label', tokenizer.batch_decode(label))
    print('gen', tokenizer.batch_decode(out))
    print()
    

