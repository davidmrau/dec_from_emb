from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from torch.utils.data import random_split
import torch
import datasets
from peft import PeftModel, PeftConfig, get_peft_model
from torch.utils.data import DataLoader
import numpy as np

def get_model(model_name):
    config = PeftConfig.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map='auto', torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base_model, model_name)
    model = model.merge_and_unload()
    model.config.pretraining_tp = 1
    model.config.use_cache = False
    model.train()
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', padding_side='right')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 128
    for p in model.parameters():
        p.requires_grad = True
    return model, tokenizer, config

model_name = 'castorini/repllama-v1-7b-lora-passage'
model, tokenizer, peft_config= get_model(model_name)

dataset = datasets.load_from_disk('kilt-100w_emb_auto_reg_pretrain_task.hf').select(range(1000))
def tokenize_function(examples, max_length=128):
    tokenized_labels = tokenizer(examples['labels'], return_tensors='pt', padding='max_length', max_length=128, truncation=True)
    examples['labels'] = tokenized_labels['input_ids']
    examples['inputs_embeds'] = np.expand_dims(np.array(examples['inputs_embeds']),1)
    return examples

dataset = dataset.map(tokenize_function, num_proc=1, batched=False)
dataset = dataset.train_test_split(test_size=0.1)

training_args = TrainingArguments(
    output_dir="./my_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    learning_rate=4e-4,
    per_device_train_batch_size=1,
    save_total_limit=1,
    logging_steps=10,  # Log every 1000 steps
    eval_steps=100,  # Evaluate on dev every 1000 steps
    remove_unused_columns=False,
    gradient_accumulation_steps=1,
)
from transformers import Trainer
class MyTrainer(Trainer):
    def compute_loss(self, model, inputs):
        labels = inputs.get("labels")
        inputs_embeds = inputs.get("inputs_embeds")
        print(inputs_embeds.shape)
        generated_ids = model.generate(inputs_embeds=inputs_embeds, max_new_tokens=128)
        print(tokenizer.batch_decode(generated_ids))
        print(tokenizer.batch_decode(labels))
        print(generated_ids)
        exit()


trainer = MyTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
)

# Training loop
trainer.train()

# # Evaluate on test set
# results = trainer.evaluate(test_dataset)
# print(results)