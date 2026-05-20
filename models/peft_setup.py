from peft import LoraConfig, get_peft_model, TaskType
from config.config import LoRAConfig
import torch

class PEFTSetup:
    def __init__(self, lora_config: LoRAConfig):
        self.lora_config = lora_config
        
    def get_lora_config(self):
        """Get LoRA configuration"""
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_config.r,
            lora_alpha=self.lora_config.lora_alpha,
            lora_dropout=self.lora_config.lora_dropout,
            target_modules=self.lora_config.target_modules
        )
    
    def apply_lora(self, model):
        """Apply LoRA to the model"""
        lora_config = self.get_lora_config()
        peft_model = get_peft_model(model, lora_config)
        return peft_model
    
    @staticmethod
    def count_trainable_parameters(model):
        """Count trainable parameters in the model"""
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        return {
            "trainable": trainable_params,
            "all": all_params,
            "percentage": 100 * trainable_params / all_params
        }