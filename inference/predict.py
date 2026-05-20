import torch
import time
from typing import Optional
from transformers import GenerationConfig
from peft import PeftModel
from config.config import GenerationConfig as GenConfig

class MedicalInference:
    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        
    def generate_response(self, 
                         dialogue: str, 
                         generation_config: Optional[GenConfig] = None) -> str:
        """Generate response for a medical query"""
        if generation_config is None:
            generation_config = GenConfig()
        
        start_time = time.time()
        
        # Create prompt
        messages = [{"role": "user", "content": dialogue}]
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        
        # Generate
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=generation_config.max_new_tokens,
                temperature=generation_config.temperature,
                do_sample=generation_config.do_sample,
                top_p=generation_config.top_p,
                repetition_penalty=generation_config.repetition_penalty
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()
        
        inference_time = time.time() - start_time
        
        return {
            "response": response,
            "inference_time": inference_time,
            "input": dialogue
        }
    
    def batch_generate(self, dialogues: list, **kwargs) -> list:
        """Generate responses for multiple dialogues"""
        return [self.generate_response(dialogue, **kwargs) for dialogue in dialogues]