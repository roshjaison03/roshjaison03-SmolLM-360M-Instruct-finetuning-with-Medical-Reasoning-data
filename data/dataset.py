from datasets import load_dataset
from typing import Dict, Any
from config.config import DatasetConfig

class MedicalDataset:
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.dataset = None
        
    def load(self):
        """Load the medical dataset"""
        self.dataset = load_dataset(
            self.config.dataset_name, 
            self.config.subset
        )
        return self.dataset
    
    def get_train_dataset(self):
        return self.dataset['train'] if self.dataset else None
    
    def get_example(self, idx: int = 0, split: str = 'train'):
        """Get a single example for inspection"""
        if self.dataset is None:
            self.load()
        return self.dataset[split][idx]