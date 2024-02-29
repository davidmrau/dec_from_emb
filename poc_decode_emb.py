from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import Trainer, TrainingArguments
import torch
import datasets
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import datasets
from modeling_llama import LlamaForCausalLM



f(t1, .... tn) -> h 

f(f) -> t1, .... tn



import os 
#model_name = 'meta-llama/Llama-2-7b-chat-hf'
# model_name = 'google/gemma-2b-it'
model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='right', torch_dtype=torch.bfloat16)
 # add tokens
tokenizer.emb_token = '<emb>'
tokenizer.add_tokens([tokenizer.emb_token], special_tokens=True) 
tokenizer.emb_token_id = tokenizer.encode(tokenizer.emb_token, add_special_tokens=False)[0]

tokenizer.agg_token = '<agg>'
tokenizer.add_tokens([tokenizer.agg_token], special_tokens=True) 
tokenizer.agg_token_id = tokenizer.encode(tokenizer.agg_token, add_special_tokens=False)[0]

tokenizer.pad_token = tokenizer.bos_token


#TODO: use aggregation token

dataset = datasets.load_from_disk('/local/calmar/rag/datasets/kilt-100w_full/').shuffle(seed=42)

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
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    target_modules=['q_proj', 'down_proj', 'gate_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj'],
    )

print(lora_config)
# get adapter
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
doc_max_length = 16



# # getting h aggregation
# t1, .... tn <agg>

# # using h  during training

# inp:    <bos><emb>      t1, .... tn <eos>
# labels: [-100] [-100]   t1, .... tn <eos>

# # inference 
# <bos><emb> 


def collate_fn(examples):
    inp = [tokenizer.emb_token + ' '.join(e['content'].split(' ')[:doc_max_length]) + tokenizer.agg_token for e in examples]
    inp = tokenizer(inp, return_tensors='pt', padding=True)


    inp_label = [tokenizer.emb_token + ' '.join(e['content'].split(' ')[:doc_max_length]) + tokenizer.eos_token for e in examples]
    inp_label = tokenizer(inp_label, return_tensors='pt', padding=True)

    label = inp_label['input_ids']
    label[label == tokenizer.emb_token_id] = -100
    label[label == tokenizer.bos_token_id] = -100
    label[label == tokenizer.pad_token_id] = -100
    inp.update({'label': label})
    return inp

os.environ["WANDB_PROJECT"] = 'dec_from_emb'

def get_emb_repr(model, inp):
    emb = model(**inp, output_hidden_states=True, is_causal=False).hidden_states[-1]
    mask = inp['input_ids'] == tokenizer.agg_token_id
    return emb[mask].unsqueeze(1)

def replace_embeddings(input_ids, inputs_embeds, insert_embeds, placeholder_id):
    insert_embeds = insert_embeds.to(inputs_embeds.device)
    # get indexes of placeholder ids
    replace_index = (input_ids == placeholder_id).nonzero(as_tuple=True)[1][0]
    inputs_embeds = torch.cat((inputs_embeds[:, :replace_index], insert_embeds, inputs_embeds[:, replace_index+1:] ), 1)
    return inputs_embeds

class MyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        label = inputs.pop('label')
        emb = get_emb_repr(model, inputs)
        inputs_embeds = model.get_input_embeddings()(input_ids)
        inputs_embeds = replace_embeddings(input_ids, inputs_embeds, emb, tokenizer.emb_token_id)
        # inputs_embeds = torch.cat((inputs_embeds[:, 0].unsqueeze(1), eos_emb, inputs_embeds[:, 2:]), 1)
        out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=label)
        return out.loss


if __name__ == "__main__": 
    out = model_name.replace('/', '_')
    training_args = TrainingArguments(
        output_dir=f"./emb_{out}",
        overwrite_output_dir=True,
        num_train_epochs=10,
        learning_rate=5e-3,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        logging_steps=10,  # Log every 1000 steps
        remove_unused_columns=False,
        gradient_accumulation_steps=1,
        report_to=None,
        save_strategy="steps",
        save_steps=10,
        do_eval=True,
    )
    trainer = MyTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=dataset,
    )
    tokenizer.save_pretrained(out)
    # Training loop
    trainer.train()
