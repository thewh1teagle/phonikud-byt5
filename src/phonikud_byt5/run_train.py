#!/usr/bin/env python3

import os
import torch
from transformers import T5ForConditionalGeneration, ByT5Tokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset

from config import TrainArgs
from utils import prepare_lines, calculate_wer_cer_metrics, log_metrics, TrainingLine


class HebrewG2PDataset(Dataset):
    def __init__(self, lines, tokenizer, max_length=512):
        self.lines = lines
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, idx):
        line = self.lines[idx]
        
        # Tokenize input (Hebrew text)
        inputs = self.tokenizer(
            line.unvocalized,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target (phonemes)
        targets = self.tokenizer(
            line.vocalized,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }


def main():
    args = TrainArgs().parse_args()
    
    print(f"🚀 Starting ByT5 Hebrew G2P Training")
    print(f"Data dir: {args.data_dir}")
    print(f"Model: {args.model_name}")
    
    # Load data
    train_lines, val_lines = prepare_lines(args)
    
    # Initialize model and tokenizer
    print(f"📦 Loading {args.model_name}...")
    tokenizer = ByT5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    
    # Set up device with fallback
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("⚠️  MPS not available, falling back to CPU")
        device = torch.device("cpu")
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print(f"🔧 Using device: {device}")
    model = model.to(device)
    
    # Create datasets
    train_dataset = HebrewG2PDataset(train_lines, tokenizer, args.max_context_length)
    val_dataset = HebrewG2PDataset(val_lines, tokenizer, args.max_context_length)
    
    # Setup training
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=args.ckpt_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to=[],
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    
    # Start training
    print("🏋️ Starting training...")
    trainer.train()
    
    # Save final model
    final_model_path = f"{args.ckpt_dir}/final_model"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    print(f"✅ Training complete! Model saved to {final_model_path}")


if __name__ == "__main__":
    main()
