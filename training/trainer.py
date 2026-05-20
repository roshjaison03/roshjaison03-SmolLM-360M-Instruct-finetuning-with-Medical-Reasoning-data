from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import PeftModel
from config.config import TrainingConfig
import torch

class ModelTrainer:
    def __init__(self, model, tokenizer, training_config: TrainingConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = training_config
        
    def get_training_args(self):
        """Get training arguments"""
        return TrainingArguments(
            output_dir=self.config.output_dir,
            auto_find_batch_size=self.config.auto_find_batch_size,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            save_strategy=self.config.save_strategy,
            evaluation_strategy=self.config.evaluation_strategy,
            logging_steps=self.config.logging_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
        )
    
    def get_data_collator(self):
        """Get data collator for causal LM"""
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
    
    def train(self, train_dataset):
        """Train the model"""
        training_args = self.get_training_args()
        data_collator = self.get_data_collator()
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        trainer.train()
        return trainer
    
    def save_model(self, path: str):
        """Save the trained model"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)