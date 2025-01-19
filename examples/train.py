import logging
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
)
from mdlm.data import RandomMaskingCollator, chunk_dataset
from mdlm.model.auto import AutoModelForMaskedDiffusionLM
from mdlm.trainer import MDLMTrainer

logging.basicConfig(level=logging.INFO)

# config
chunk_size = 128


# 0) Choose base model to fintune
masked_model = "google/bert_uncased_L-2_H-128_A-2" # bert-tiny (4.4M params)
# masked_model = "bert-base-uncased" # bert-base (110M params)

# 1) Load the dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# 2) Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(masked_model)

# 3) Tokenize
def _tokenize_function(examples):
    return tokenizer(examples["text"])
dataset = dataset.map(
    _tokenize_function, 
    batched=True,
    remove_columns=["text"]
)

# 4) Chunk
dataset = dataset.map(
    chunk_dataset, 
    batched=True, 
    fn_kwargs={'chunk_size': chunk_size}
)

# 5) Create custom collator
data_collator = RandomMaskingCollator(
    tokenizer=tokenizer,
    min_mlm_prob=0.0,
    max_mlm_prob=1.0
    # min_mlm_prob=0.10,
    # max_mlm_prob=0.85    
)

# 6) Initialize model (e.g., BERT for MLM)
model = AutoModelForMaskedDiffusionLM.from_pretrained(masked_model)

# 7) Setup Trainer
training_args = TrainingArguments(
    output_dir="./mdlm-output",
    evaluation_strategy="no",
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_steps=10,
)

trainer = MDLMTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 8) Train!
trainer.train()

# 9) Save pre-trained model
model.save_pretrained('./masked-diffusion-language-model')