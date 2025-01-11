import logging
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from my_collator import PerSequenceRandomMaskingCollatorNoRandom  # <— Your custom collator code

logging.basicConfig(level=logging.INFO)

# config
chunk_size = 128

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=1024)  
    # or no max_length, if you plan to rely solely on chunking

if __name__ == "__main__":
    # 1) Load your dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")

    # 2) Initialize your tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # 3) Tokenize
    dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    # 4) Chunk
    dataset = dataset.map(chunk_dataset, batched=True)
    
    # 5) Create custom collator
    data_collator = PerSequenceRandomMaskingCollatorNoRandom(
        tokenizer=tokenizer,
        min_mlm_prob=0.05,
        max_mlm_prob=0.25
    )

    # 6) Initialize your model (e.g., BERT for MLM)
    from transformers import AutoModelForMaskedLM
    model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

    # 7) Setup Trainer
    training_args = TrainingArguments(
        output_dir="./test_chunked_no_random",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        logging_steps=10
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # 8) Train!
    trainer.train()

    # 9) Save pre-trained model
    model.save_pretrained('./saved-diffusion-model')