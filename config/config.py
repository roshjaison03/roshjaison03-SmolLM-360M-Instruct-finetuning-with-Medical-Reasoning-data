import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    model_name: str = "HuggingFaceTB/SmolLM-360M-Instruct"
    revision: str = "v0.1"
    max_length: int = 1024
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

@dataclass
class LoRAConfig:
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: list = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

@dataclass
class TrainingConfig:
    output_dir: str = "./outputs"
    learning_rate: float = 1e-3
    num_train_epochs: int = 1
    auto_find_batch_size: bool = True
    save_strategy: str = "epoch"
    evaluation_strategy: str = "no"
    logging_steps: int = 10
    save_total_limit: int = 2
    load_best_model_at_end: bool = False

@dataclass
class DatasetConfig:
    dataset_name: str = "FreedomIntelligence/medical-o1-reasoning-SFT"
    subset: str = "en"
    system_prompt: str = "You are an expert assistant. Think step by step and provide a clear reasoning before your final answer."

@dataclass
class GenerationConfig:
    max_new_tokens: int = 200
    temperature: float = 0.7
    do_sample: bool = True
    top_p: float = 0.9
    repetition_penalty: float = 1.2