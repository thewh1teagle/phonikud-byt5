from typing import Literal, Optional
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
    num_epochs: int = 10
    val_split: float = 0.01  # 1% for validation
    split_seed: int = 42
    device: Literal["mps", "cuda", "cpu"] = "cuda"  # Device to use: "mps", "cuda", "cpu"
    
    # Evaluation & Saving
    eval_steps: int = 500  # How often to evaluate (and save best/last models)
    save_steps: int = 1000  # How often to save trainer checkpoints
    logging_steps: int = 100  # How often to log training metrics
    
    # Wandb configuration (for experiment tracking)
    wandb_entity: str = "Phonikud"  # Team or username for Weights & Biases
    wandb_project: str = "phonikud"  # Project name for Weights & Biases
    wandb_mode: str = "online"  # Wandb mode: 'online', 'offline', or 'disabled' (default: offline for local use)
    
    # Optional limits for testing
    max_lines: Optional[int] = None  # Limit dataset size for testing
