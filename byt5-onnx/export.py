#!/usr/bin/env python3

from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import ByT5Tokenizer
from pathlib import Path
from tap import Tap


class ExportArgs(Tap):
    model_path: str = "../checkpoints/best_model"  # Path to trained model directory
    output_dir: str = "./onnx_model"  # Output directory for ONNX model


def main():
    """Export ByT5 model to ONNX using HuggingFace Optimum"""
    args = ExportArgs().parse_args()
    
    print(f"🚀 Exporting ByT5 model to ONNX using HuggingFace Optimum")
    print(f"   Model: {args.model_path}")
    print(f"   Output: {args.output_dir}")
    
    # Check if model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Model path not found: {model_path}")
        return
    
    # Check for config.json
    config_path = model_path / "config.json"
    if not config_path.exists():
        print(f"❌ config.json not found in {model_path}")
        return
    
    try:
        print(f"📦 Loading and converting model...")
        
        # Load and convert the model using Optimum
        # This automatically converts the PyTorch model to ONNX
        ort_model = ORTModelForSeq2SeqLM.from_pretrained(
            str(model_path),
            export=True,  # This triggers the conversion
        )
        
        print(f"✅ Model converted successfully")
        
        # Save the ONNX model
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Saving ONNX model to {output_path}...")
        ort_model.save_pretrained(str(output_path))
        
        # Also save the tokenizer
        print(f"📝 Saving tokenizer...")
        tokenizer = ByT5Tokenizer.from_pretrained(str(model_path))
        tokenizer.save_pretrained(str(output_path))
        
        print(f"✅ Export completed successfully!")
        print(f"   ONNX model saved to: {output_path}")
        
        # Show file sizes
        onnx_files = list(output_path.glob("*.onnx"))
        if onnx_files:
            print(f"📁 Generated ONNX files:")
            for onnx_file in onnx_files:
                size_mb = onnx_file.stat().st_size / (1024 * 1024)
                print(f"   - {onnx_file.name}: {size_mb:.1f} MB")
        
        print(f"\n🎉 Ready to use!")
        print(f"   Run: python predict.py --model_path {output_path}")
        
    except Exception as e:
        print(f"❌ Error during export: {e}")
        print(f"   Make sure the model is a valid ByT5 model")
        print(f"   Try: pip install --upgrade optimum[onnxruntime]")


if __name__ == "__main__":
    main()
