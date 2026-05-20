from typing import Dict, List
import torch
from transformers import PreTrainedTokenizer
from config.config import DatasetConfig, ModelConfig

class TokenizationProcessor:
    def __init__(self, tokenizer: PreTrainedTokenizer, dataset_config: DatasetConfig, model_config: ModelConfig):
        self.tokenizer = tokenizer
        self.dataset_config = dataset_config
        self.model_config = model_config
        
    def tokenize_function(self, examples: Dict) -> Dict:
        """Tokenize examples for training"""
        MAX_LENGTH = self.model_config.max_length
        
        inputs = []
        attention_masks = []
        all_labels = []
        
        for question, cot, response in zip(
            examples["Question"], 
            examples["Complex_CoT"], 
            examples["Response"]
        ):
            # Create conversation with system prompt
            messages = [
                {"role": "system", "content": self.dataset_config.system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": cot},
                {"role": "assistant", "content": response}
            ]
            
            # Apply chat template
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                truncation=False,
                add_generation_prompt=False
            )
            
            # Pad/truncate
            if len(formatted) > MAX_LENGTH:
                formatted = formatted[:MAX_LENGTH]
            else:
                formatted = formatted + [self.tokenizer.pad_token_id] * (MAX_LENGTH - len(formatted))
            
            # Create attention mask
            attention_mask = [1 if token != self.tokenizer.pad_token_id else 0 for token in formatted]
            
            # Create labels (only train on assistant responses)
            labels = [-100] * MAX_LENGTH
            
            # Find assistant start position
            context_messages = messages[:2]
            context_encoded = self.tokenizer.apply_chat_template(
                context_messages,
                tokenize=True,
                add_generation_prompt=True
            )
            assistant_start_idx = len(context_encoded)
            
            # Tokenize assistant content
            assistant_content = f"{cot}\n\nFinal Answer: {response}"
            assistant_encoded = self.tokenizer.encode(assistant_content, add_special_tokens=False)
            
            # Fill labels
            for i, token in enumerate(assistant_encoded):
                if assistant_start_idx + i < MAX_LENGTH:
                    labels[assistant_start_idx + i] = token
            
            inputs.append(formatted)
            attention_masks.append(attention_mask)
            all_labels.append(labels)
        
        return {
            "input_ids": inputs,
            "attention_mask": attention_masks,
            "labels": all_labels
        }