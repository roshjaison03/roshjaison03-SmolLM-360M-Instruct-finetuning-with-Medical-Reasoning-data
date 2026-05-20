import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config.config import ModelConfig

class ModelLoader:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the base model"""
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            revision=self.config.revision
        ).to(self.config.device)
        return self.model
    
    def load_tokenizer(self):
        """Load the tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        return self.tokenizer
    
    def load_both(self):
        """Load both model and tokenizer"""
        self.load_model()
        self.load_tokenizer()
        return self.model, self.tokenizer