#!/usr/bin/env python3

from typing import Literal
import torch
from transformers import T5ForConditionalGeneration, ByT5Tokenizer
from tap import Tap


class PredictArgs(Tap):
    model_path: str = "checkpoints/final_model"  # Path to trained model directory  
    device: Literal["mps", "cuda", "cpu"] = "cuda"  # Device to use: "mps", "cuda", "cpu"
    max_context_length: int = 512


def predict_text(model, tokenizer, text, max_length=512, device=None):
    """Predict phonemes for a Hebrew text"""
    # Tokenize input
    inputs = tokenizer(
        text,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    
    # Move inputs to device if specified
    if device is not None:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate prediction
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            num_beams=4,
            do_sample=False,
        )
    
    # Decode prediction
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prediction


def main():
    args = PredictArgs().parse_args()
    
    # Hebrew text to predict
    hebrew_text = "שלום עולם!"
    
    print(f"📦 Loading model from {args.model_path}...")
    
    # Load model and tokenizer
    tokenizer = ByT5Tokenizer.from_pretrained(args.model_path)
    model = T5ForConditionalGeneration.from_pretrained(args.model_path)
    
    # Set up device with fallback
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("⚠️  MPS not available, falling back to CPU")
        args.device = "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU") 
        args.device = "cpu"
    
    device_obj = torch.device(args.device)
    print(f"🔧 Using device: {device_obj}")
    model = model.to(device_obj)
    model.eval()
    
    print(f"🔮 Predicting phonemes for: {hebrew_text}")
    
    # Predict
    prediction = predict_text(model, tokenizer, hebrew_text, args.max_context_length, device_obj)
    
    print(f"\n📝 Results:")
    print(f"Input:      {hebrew_text}")
    print(f"Phonemes:   {prediction}")


if __name__ == "__main__":
    main()
