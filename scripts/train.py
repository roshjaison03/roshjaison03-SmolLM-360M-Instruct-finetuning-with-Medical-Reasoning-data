#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import ModelConfig, LoRAConfig, TrainingConfig, DatasetConfig
from data.dataset import MedicalDataset
from models.model_loader import ModelLoader
from models.peft_setup import PEFTSetup
from training.tokenization import TokenizationProcessor
from training.trainer import ModelTrainer
from utils.helpers import print_model_summary, cleanup_directory

def main():
    # Load configurations
    model_config = ModelConfig()
    lora_config = LoRAConfig()
    training_config = TrainingConfig()
    dataset_config = DatasetConfig()
    
    # Clean up previous outputs
    cleanup_directory(training_config.output_dir)
    cleanup_directory("./peft-response")
    
    # Load dataset
    print("Loading dataset...")
    dataset = MedicalDataset(dataset_config)
    dataset.load()
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model_loader = ModelLoader(model_config)
    model, tokenizer = model_loader.load_both()
    
    # Print initial model summary
    print("Initial model:")
    print_model_summary(model)
    
    # Apply LoRA
    print("\nApplying LoRA...")
    peft_setup = PEFTSetup(lora_config)
    peft_model = peft_setup.apply_lora(model)
    print(peft_setup.count_trainable_parameters(peft_model))
    
    # Prepare tokenized dataset
    print("\nTokenizing dataset...")
    tokenization_processor = TokenizationProcessor(tokenizer, dataset_config, model_config)
    tokenized_datasets = dataset.dataset.map(
        tokenization_processor.tokenize_function,
        batched=True,
        remove_columns=['Question', 'Complex_CoT', 'Response']
    )
    
    # Train
    print("\nStarting training...")
    trainer = ModelTrainer(peft_model, tokenizer, training_config)
    trainer.train(tokenized_datasets['train'])
    
    # Save model
    print("\nSaving model...")
    trainer.save_model("./peft-response")
    print("Training completed successfully!")

if __name__ == "__main__":
    main()