from typing import Optional
from tap import Tap


class TrainArgs(Tap):
    # Data
    data_dir: str = "data" # Directory containing your 5M text files
    ckpt_dir: str = "checkpoints"  # Where to save model checkpoints
    
    # Model
    model_name: str = "google/byt5-small"  # ByT5 model to use
    max_context_length: int = 512  # Max sequence length
    
    # Training
    batch_size: int = 8
    learning_rate: float = 5e-5
    num_epochs: int = 3
    val_split: float = 0.01  # 1% for validation
    split_seed: int = 42
    device: str = "mps"  # Device to use: "mps", "cuda", "cpu"
    
    # Optional limits for testing
    max_lines: Optional[int] = None  # Limit dataset size for testing
