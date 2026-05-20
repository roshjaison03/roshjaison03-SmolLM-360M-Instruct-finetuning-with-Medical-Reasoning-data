#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from peft import PeftModel
from config.config import ModelConfig, GenerationConfig
from models.model_loader import ModelLoader
from inference.predict import MedicalInference

def main():
    # Load configs
    model_config = ModelConfig()
    gen_config = GenerationConfig()
    
    # Load base model and tokenizer
    print("Loading base model...")
    model_loader = ModelLoader(model_config)
    base_model, tokenizer = model_loader.load_both()
    
    # Load fine-tuned PEFT model
    print("Loading fine-tuned PEFT model...")
    peft_model = PeftModel.from_pretrained(base_model, "./peft-response")
    
    # Create inference pipeline
    print("Starting inference...")
    inference = MedicalInference(peft_model, tokenizer, device=model_config.device)
    
    # Example queries
    test_queries = [
        "Hello Doctor, I've been experiencing a persistent headache along with stomach discomfort since today. I also feel a bit fatigued and uneasy overall.",
        "I have a fever and cough for 3 days. Should I be worried?",
        "What are the symptoms of diabetes?"
    ]
    
    # Generate responses
    for query in test_queries:
        print("\n" + "="*60)
        result = inference.generate_response(query, gen_config)
        print(f"Query: {result['input']}")
        print(f"Response: {result['response']}")
        print(f"Time: {result['inference_time']:.2f}s")
        print("="*60)

if __name__ == "__main__":
    main()