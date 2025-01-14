import logging
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from mdlm.data import RandomMaskingCollator, chunk_dataset
from mdlm.model.auto import AutoModelForMaskedDiffusionLM
from mdlm.trainer import MDLMTrainer
from mdlm.eval import compute_metrics

logging.basicConfig(level=logging.INFO)

# config
chunk_size = 128


if __name__ == "__main__":
    # 1) Load your dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")

    # 2) Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # 3) Tokenize
    def _tokenize_function(examples):
        return tokenizer(examples["text"])
    dataset = dataset.map(
        _tokenize_function, 
        batched=True,
        remove_columns=["text", "label"]
    )

    # 4) Chunk
    dataset = dataset.map(chunk_dataset, batched=True)
    
    # 5) Create custom collator
    data_collator = RandomMaskingCollator(
        tokenizer=tokenizer,
        min_mlm_prob=0.10,
        max_mlm_prob=0.85
    )

    # 6) Initialize model (e.g., BERT for MLM)
    model = AutoModelForMaskedDiffusionLM.from_pretrained("bert-base-uncased")

    # 7) Setup Trainer
    training_args = TrainingArguments(
        output_dir="./mdlm-output",
        evaluation_strategy="steps",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_steps=10,
        logging_steps=10
    )
    trainer = MDLMTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # 8) Train!
    trainer.train()

    # 9) Save pre-trained model
    model.save_pretrained('./masked-diffusion-language-model')