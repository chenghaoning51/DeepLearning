"""
Configuration Manager for Code Summarization Experiment
配置管理器 - 负责加载、验证和管理实验配置参数
"""

import os
import yaml
import json
import shutil
from typing import Any, Dict, Optional
from pathlib import Path
from datetime import datetime


class ConfigError(Exception):
    """Configuration related errors"""
    pass


class ConfigManager:
    """
    配置管理器
    
    职责：
    - 加载YAML配置文件
    - 验证配置参数的有效性
    - 支持命令行参数覆盖
    - 保存配置文件副本以确保实验可复现
    """
    
    def __init__(self, config_path: str):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = {}
        self._load_config()
        self._validate_config()
    
    def _load_config(self) -> None:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            raise ConfigError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML format in {self.config_path}: {e}")
        except Exception as e:
            raise ConfigError(f"Failed to load configuration: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值，支持嵌套键（如 'model.learning_rate'）
        
        Args:
            key: 配置键，支持点分隔的嵌套键
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        设置配置值，支持嵌套键
        
        Args:
            key: 配置键
            value: 配置值
        """
        keys = key.split('.')
        config = self.config
        
        # 导航到最后一级
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # 设置值
        config[keys[-1]] = value
    
    def update_from_args(self, args: Dict[str, Any]) -> None:
        """
        从命令行参数更新配置
        
        Args:
            args: 命令行参数字典
        """
        # 映射命令行参数到配置键
        arg_mapping = {
            'learning_rate': 'model.learning_rate',
            'batch_size': 'model.batch_size',
            'num_epochs': 'model.num_epochs',
            'output_dir': 'experiment.output_dir',
            'model_name': 'model.name',
            'log_level': 'logging.log_level',
        }
        
        for arg_key, config_key in arg_mapping.items():
            if arg_key in args and args[arg_key] is not None:
                self.set(config_key, args[arg_key])
    
    def _validate_config(self) -> None:
        """验证配置参数的有效性"""
        validators = {
            'model.learning_rate': self._validate_learning_rate,
            'model.batch_size': self._validate_batch_size,
            'model.num_epochs': self._validate_num_epochs,
            'data.max_source_length': self._validate_max_length,
            'data.max_target_length': self._validate_max_length,
            'experiment.seed': self._validate_seed,
            'evaluation.beam_size': self._validate_beam_size,
            'model.name': self._validate_model_name,
            'experiment.output_dir': self._validate_output_dir,
        }
        
        errors = []
        for key, validator in validators.items():
            try:
                value = self.get(key)
                if value is not None:
                    validator(value, key)
            except ValueError as e:
                errors.append(str(e))
        
        if errors:
            raise ConfigError(f"Configuration validation failed:\n" + "\n".join(errors))
    
    def validate(self) -> bool:
        """
        公开的验证方法，用于外部调用
        
        Returns:
            True if validation passes
            
        Raises:
            ConfigError if validation fails
        """
        self._validate_config()
        return True
    
    def _validate_learning_rate(self, value: Any, key: str) -> None:
        """验证学习率"""
        if not isinstance(value, (int, float)) or value <= 0 or value > 1:
            raise ValueError(
                f"Invalid {key}: {value}. Must be a positive number less than or equal to 1.\n"
                f"Valid range: (0, 1]. Common values: 1e-5, 5e-5, 1e-4, 5e-4"
            )
    
    def _validate_batch_size(self, value: Any, key: str) -> None:
        """验证批次大小"""
        if not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"Invalid {key}: {value}. Must be a positive integer.\n"
                f"Valid range: [1, ∞). Common values: 8, 16, 32, 64"
            )
    
    def _validate_num_epochs(self, value: Any, key: str) -> None:
        """验证训练轮数"""
        if not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"Invalid {key}: {value}. Must be a positive integer.\n"
                f"Valid range: [1, ∞). Common values: 3, 5, 10"
            )
    
    def _validate_max_length(self, value: Any, key: str) -> None:
        """验证最大长度"""
        if not isinstance(value, int) or value < 10 or value > 1024:
            raise ValueError(
                f"Invalid {key}: {value}. Must be an integer between 10 and 1024.\n"
                f"Valid range: [10, 1024]. Common values: 128, 256, 512"
            )
    
    def _validate_seed(self, value: Any, key: str) -> None:
        """验证随机种子"""
        if not isinstance(value, int) or value < 0:
            raise ValueError(
                f"Invalid {key}: {value}. Must be a non-negative integer.\n"
                f"Valid range: [0, ∞). Common values: 42, 0, 123"
            )
    
    def _validate_beam_size(self, value: Any, key: str) -> None:
        """验证beam size"""
        if not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"Invalid {key}: {value}. Must be a positive integer.\n"
                f"Valid range: [1, ∞). Common values: 1, 3, 5, 10"
            )
    
    def _validate_model_name(self, value: Any, key: str) -> None:
        """验证模型名称"""
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Invalid {key}: {value}. Must be a non-empty string.\n"
                f"Valid examples: 'Salesforce/codet5-small', 'microsoft/codebert-base'"
            )
    
    def _validate_output_dir(self, value: Any, key: str) -> None:
        """验证输出目录"""
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Invalid {key}: {value}. Must be a non-empty string path.\n"
                f"Valid example: './outputs'"
            )
    
    def save_config(self, output_path: str) -> None:
        """
        保存配置文件副本，包含元数据以确保实验可复现
        
        Args:
            output_path: 输出路径
        """
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 添加元数据
            config_with_metadata = self.config.copy()
            config_with_metadata['_metadata'] = {
                'saved_at': datetime.now().isoformat(),
                'saved_to': str(Path(output_path).absolute()),
                'original_config_path': self.config_path,
            }
            
            # 保存配置
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_with_metadata, f, default_flow_style=False, allow_unicode=True)
                
        except Exception as e:
            raise ConfigError(f"Failed to save configuration to {output_path}: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """返回配置字典"""
        return self.config.copy()
    
    def get_parameter_info(self) -> str:
        """
        获取所有配置参数的信息，包括有效范围和示例
        
        Returns:
            格式化的参数信息字符串
        """
        info = """
Configuration Parameters Information:

Model Parameters:
  - model.name: Model name from Hugging Face
    Valid examples: 'Salesforce/codet5-small', 'microsoft/codebert-base'
  
  - model.learning_rate: Learning rate for training
    Valid range: (0, 1]
    Common values: 1e-5, 5e-5, 1e-4, 5e-4
  
  - model.batch_size: Batch size for training
    Valid range: [1, ∞)
    Common values: 8, 16, 32, 64
  
  - model.num_epochs: Number of training epochs
    Valid range: [1, ∞)
    Common values: 3, 5, 10

Data Parameters:
  - data.max_source_length: Maximum length for source code
    Valid range: [10, 1024]
    Common values: 128, 256, 512
  
  - data.max_target_length: Maximum length for target summary
    Valid range: [10, 1024]
    Common values: 64, 128, 256

Experiment Parameters:
  - experiment.seed: Random seed for reproducibility
    Valid range: [0, ∞)
    Common values: 42, 0, 123
  
  - experiment.output_dir: Output directory path
    Valid example: './outputs'

Evaluation Parameters:
  - evaluation.beam_size: Beam size for generation
    Valid range: [1, ∞)
    Common values: 1, 3, 5, 10
"""
        return info
    
    def __str__(self) -> str:
        """字符串表示"""
        return yaml.dump(self.config, default_flow_style=False, allow_unicode=True)


def get_nested_value(config: Dict, key: str) -> Any:
    """
    获取嵌套字典中的值
    
    Args:
        config: 配置字典
        key: 点分隔的键
        
    Returns:
        配置值
    """
    keys = key.split('.')
    value = config
    
    for k in keys:
        value = value[k]
    
    return value