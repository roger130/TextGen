import torch
from dataclasses import dataclass

@dataclass
class Config:
    # Model
    model_name: str = "gpt2"
    max_length: int = 512
    
    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: list = None
    
    # Training
    batch_size: int = 8
    learning_rate: float = 5e-4
    num_epochs: int = 5
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Early stopping
    patience: int = 2
    min_delta: float = 0.01
    
    # Data
    train_size: int = 20000
    val_size: int = 2000
    
    # Paths
    output_dir: str = "./outputs"
    cache_dir: str = "./cache"
    
    # Device
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["c_attn", "c_proj"]
        
        if self.device == "mps":
            self.batch_size = min(self.batch_size, 4)