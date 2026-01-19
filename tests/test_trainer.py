"""
Unit tests for ModelTrainer
"""

import pytest
import sys
import os
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.manager import ConfigManager
from models.trainer import ModelTrainer, ModelError


class TestModelTrainer:
    """Test cases for ModelTrainer"""
    
    def test_device_setup(self):
        """Test device setup (CPU/GPU detection)"""
        config = ConfigManager('config/default.yaml')
        trainer = ModelTrainer(config)
        
        # Device should be set
        assert trainer.device is not None
        
        # Device should be either cuda or cpu
        assert str(trainer.device) in ['cuda', 'cpu']
        
        # If CUDA is available, device should be cuda
        if torch.cuda.is_available():
            assert str(trainer.device) == 'cuda'
        else:
            assert str(trainer.device) == 'cpu'
    
    def test_model_name_from_config(self):
        """Test that model name is correctly loaded from config"""
        config = ConfigManager('config/default.yaml')
        trainer = ModelTrainer(config)
        
        # Model name should match config
        expected_name = config.get("model.name", "Salesforce/codet5-small")
        assert trainer.model_name == expected_name
    
    def test_model_name_override(self):
        """Test that model name can be overridden in constructor"""
        config = ConfigManager('config/default.yaml')
        custom_name = "microsoft/codebert-base"
        trainer = ModelTrainer(config, model_name=custom_name)
        
        # Model name should be the custom one
        assert trainer.model_name == custom_name
    
    def test_get_model_info_before_loading(self):
        """Test get_model_info returns empty dict before model is loaded"""
        config = ConfigManager('config/default.yaml')
        trainer = ModelTrainer(config)
        
        info = trainer.get_model_info()
        assert info == {}
    
    def test_training_without_setup_raises_error(self):
        """Test that training without setup raises ModelError"""
        config = ConfigManager('config/default.yaml')
        trainer = ModelTrainer(config)
        
        with pytest.raises(ModelError, match="Trainer must be setup before training"):
            trainer.train()
    
    def test_save_checkpoint_without_model_raises_error(self):
        """Test that saving checkpoint without model raises ModelError"""
        config = ConfigManager('config/default.yaml')
        trainer = ModelTrainer(config)
        
        with pytest.raises(ModelError, match="Model and tokenizer must be loaded"):
            trainer.save_checkpoint("./test_checkpoint")
    
    def test_setup_training_without_model_raises_error(self):
        """Test that setup_training without model raises ModelError"""
        config = ConfigManager('config/default.yaml')
        trainer = ModelTrainer(config)
        
        # Create dummy datasets
        from datasets import Dataset
        dummy_data = [{'input_ids': [1, 2, 3], 'attention_mask': [1, 1, 1], 'labels': [1, 2, 3]}]
        train_dataset = Dataset.from_list(dummy_data)
        val_dataset = Dataset.from_list(dummy_data)
        
        with pytest.raises(ModelError, match="Model and tokenizer must be loaded"):
            trainer.setup_training(train_dataset, val_dataset)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
