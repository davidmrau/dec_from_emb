from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import Trainer, TrainingArguments
import torch
import datasets
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import datasets
from modeling_llama import LlamaForCausalLM

#f(t1, .... tn) -> h 
#f(f) -> t1, .... tn
import os 
#model_name = 'meta-llama/Llama-2-7b-chat-hf'
# model_name = 'google/gemma-2b-it'
model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='right', torch_dtype=torch.bfloat16)
tokenizer.pad_token = tokenizer.bos_token
doc_max_length = 100
 # add tokens
tokenizer.emb_token1 = '<emb1>'
tokenizer.add_tokens([tokenizer.emb_token1], special_tokens=True) 
tokenizer.emb_token_id1 = tokenizer.encode(tokenizer.emb_token1, add_special_tokens=False)[0]
tokenizer.emb_token2 = '<emb2>'
tokenizer.add_tokens([tokenizer.emb_token2], special_tokens=True) 
tokenizer.emb_token_id2 = tokenizer.encode(tokenizer.emb_token2, add_special_tokens=False)[0]
tokenizer.emb_token3 = '<emb3>'
tokenizer.add_tokens([tokenizer.emb_token3], special_tokens=True) 
tokenizer.emb_token_id3 = tokenizer.encode(tokenizer.emb_token3, add_special_tokens=False)[0]
tokenizer.emb_token4 = '<emb4>'
tokenizer.add_tokens([tokenizer.emb_token4], special_tokens=True) 
tokenizer.emb_token_id4 = tokenizer.encode(tokenizer.emb_token4, add_special_tokens=False)[0]

tokenizer.agg_token1 = '<agg1>'
tokenizer.add_tokens([tokenizer.agg_token1], special_tokens=True) 
tokenizer.agg_token_id1 = tokenizer.encode(tokenizer.agg_token1, add_special_tokens=False)[0]

tokenizer.agg_token2 = '<agg2>'
tokenizer.add_tokens([tokenizer.agg_token2], special_tokens=True) 
tokenizer.agg_token_id2 = tokenizer.encode(tokenizer.agg_token2, add_special_tokens=False)[0]

tokenizer.agg_token3 = '<agg3>'
tokenizer.add_tokens([tokenizer.agg_token3], special_tokens=True) 
tokenizer.agg_token_id3 = tokenizer.encode(tokenizer.agg_token3, add_special_tokens=False)[0]

tokenizer.agg_token4 = '<agg4>'
tokenizer.add_tokens([tokenizer.agg_token4], special_tokens=True) 
tokenizer.agg_token_id4 = tokenizer.encode(tokenizer.agg_token4, add_special_tokens=False)[0]


print(tokenizer)

#TODO: use aggregation token

dataset = datasets.load_from_disk('/projects/0/gusr0546/research/RAG/datasets/kilt-100w_full/').shuffle(seed=42)

dataset = dataset.train_test_split(test_size=200)
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


# # getting h aggregation
# t1, .... tn <agg>

# # using h  during training

# inp:    <bos><emb>      t1, .... tn <eos>
# labels: [-100] [-100]   t1, .... tn <eos>

# # inference 
# <bos><emb> 


def collate_fn(examples):
    inp = [tokenizer.emb_token1 + tokenizer.emb_token2 + tokenizer.emb_token3 + ' '.join(e['content'].split(' ')[:doc_max_length]) + tokenizer.agg_token1 + tokenizer.agg_token2 + tokenizer.agg_token3 + tokenizer.eos_token for e in examples]
    inp = tokenizer(inp, return_tensors='pt', padding=True)


    inp_label = [tokenizer.emb_token1 + tokenizer.emb_token2 + tokenizer.emb_token3  + ' '.join(e['content'].split(' ')[:doc_max_length]) + tokenizer.eos_token for e in examples]
    inp_label = tokenizer(inp_label, return_tensors='pt', padding=True)

    label = inp_label['input_ids'].clone()
    label[label == tokenizer.emb_token_id1] = -100
    label[label == tokenizer.emb_token_id2] = -100
    label[label == tokenizer.emb_token_id3] = -100
    label[label == tokenizer.emb_token_id4] = -100
    label[label == tokenizer.bos_token_id] = -100
    label[label == tokenizer.pad_token_id] = -100
    inp.update({'inp_label': inp_label})
    return inp

os.environ["WANDB_PROJECT"] = 'dec_from_emb'

def get_emb_repr(model, inp, token_id):
    emb = model(**inp, output_hidden_states=True, is_causal=False).hidden_states[-1]
    mask = inp['input_ids'] == token_id
    return emb[mask].unsqueeze(1)

def replace_embeddings(input_ids, inputs_embeds, insert_embeds, placeholder_id):
    insert_embeds = insert_embeds.to(inputs_embeds.device)
    # get indexes of placeholder ids
    replace_index = (input_ids == placeholder_id).nonzero(as_tuple=True)[1][0]
    inputs_embeds = torch.cat((inputs_embeds[:, :replace_index], insert_embeds, inputs_embeds[:, replace_index+1:] ), 1)
    return inputs_embeds

def replace_all(model, inputs, inputs_embeds):
    emb1 = get_emb_repr(model, inputs, tokenizer.agg_token_id1)
    inputs_embeds = replace_embeddings(inputs['input_ids'], inputs_embeds, emb1, tokenizer.emb_token_id1)
    emb2 = get_emb_repr(model, inputs, tokenizer.agg_token_id2)
    inputs_embeds = replace_embeddings(inputs['input_ids'], inputs_embeds, emb2, tokenizer.emb_token_id2)
    emb3 = get_emb_repr(model, inputs, tokenizer.agg_token_id3)
    inputs_embeds = replace_embeddings(inputs['input_ids'], inputs_embeds, emb3, tokenizer.emb_token_id3)
    emb4 = get_emb_repr(model, inputs, tokenizer.agg_token_id4)
    inputs_embeds = replace_embeddings(inputs['input_ids'], inputs_embeds, emb4, tokenizer.emb_token_id4)
    return inputs_embeds


class MyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        label = inputs.pop('inp_label')
        inputs_embeds = model.get_input_embeddings()(label['input_ids'])
        inputs_embeds = replace_all(model, inputs, inputs_embeds)
        # inputs_embeds = torch.cat((inputs_embeds[:, 0].unsqueeze(1), eos_emb, inputs_embeds[:, 2:]), 1)
        out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=label['input_ids'])
        return out.loss

    def prediction_step(self, model, inputs, prediction_loss_only=None, ignore_keys=None, **kwargs):
        with torch.no_grad():
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            label = inputs.pop('inp_label')
            inputs_embeds = model.get_input_embeddings()(label['input_ids'])
            inputs_embeds = replace_all(model, inputs, inputs_embeds)
            # inputs_embeds = torch.cat((inputs_embeds[:, 0].unsqueeze(1), eos_emb, inputs_embeds[:, 2:]), 1)
            out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=label['input_ids'])
            return out.loss, out.logits, label['input_ids']
        

if __name__ == "__main__": 
    out = model_name.replace('/', '_')
    training_args = TrainingArguments(
        output_dir=f"./emb_{out}_{doc_max_length}",
        overwrite_output_dir=True,
        num_train_epochs=1,
        learning_rate=1e-4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        logging_steps=10,  # Log every 1000 steps
        remove_unused_columns=False,
        gradient_accumulation_steps=1,
        evaluation_strategy='steps',
        report_to=None,
        save_strategy="steps",
        save_steps=10,
        warmup_steps=20,
        do_eval=True,
    )
    trainer = MyTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
    )
    # Training loop
    trainer.train()
