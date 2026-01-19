"""
Logging System for Code Summarization Experiment
日志系统 - 提供统一的日志记录功能

Requirements:
- 7.1: 记录所有关键操作的时间戳和状态信息
- 7.2: 记录完整的错误堆栈信息
- 7.4: 同时输出到控制台和日志文件
- 7.5: 生成实验摘要报告
"""

import os
import sys
import logging
import datetime
import traceback
import json
import psutil
from typing import Optional, Dict, Any, List
from pathlib import Path


class ExperimentMetrics:
    """
    实验指标记录器
    
    职责：
    - 记录实验过程中的关键指标
    - 跟踪资源使用情况
    - 生成实验摘要报告
    """
    
    def __init__(self):
        self.start_time = datetime.datetime.now()
        self.end_time = None
        self.metrics = {}
        self.operations = []
        self.errors = []
        self.resource_snapshots = []
        
    def record_operation(self, operation: str, status: str, details: Optional[Dict] = None):
        """记录操作"""
        self.operations.append({
            'timestamp': datetime.datetime.now().isoformat(),
            'operation': operation,
            'status': status,
            'details': details or {}
        })
    
    def record_error(self, error_type: str, error_msg: str, context: str = ""):
        """记录错误"""
        self.errors.append({
            'timestamp': datetime.datetime.now().isoformat(),
            'error_type': error_type,
            'error_message': error_msg,
            'context': context
        })
    
    def record_metric(self, metric_name: str, value: Any):
        """记录指标"""
        self.metrics[metric_name] = value
    
    def snapshot_resources(self):
        """记录资源使用快照"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            snapshot = {
                'timestamp': datetime.datetime.now().isoformat(),
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_rss_mb': memory_info.rss / 1024 / 1024,
                'memory_percent': process.memory_percent(),
            }
            
            # 如果有GPU，记录GPU使用情况
            try:
                import torch
                if torch.cuda.is_available():
                    snapshot['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
                    snapshot['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
            except ImportError:
                pass
            
            self.resource_snapshots.append(snapshot)
        except Exception as e:
            # 如果资源监控失败，不影响主流程
            pass
    
    def finalize(self):
        """完成实验记录"""
        self.end_time = datetime.datetime.now()
        self.snapshot_resources()
    
    def get_summary(self) -> Dict[str, Any]:
        """获取实验摘要"""
        duration = (self.end_time or datetime.datetime.now()) - self.start_time
        
        summary = {
            'experiment_info': {
                'start_time': self.start_time.isoformat(),
                'end_time': self.end_time.isoformat() if self.end_time else None,
                'duration_seconds': duration.total_seconds(),
                'duration_formatted': str(duration)
            },
            'metrics': self.metrics,
            'operations_count': len(self.operations),
            'errors_count': len(self.errors),
            'operations': self.operations,
            'errors': self.errors
        }
        
        # 添加资源使用统计
        if self.resource_snapshots:
            summary['resource_usage'] = {
                'snapshots_count': len(self.resource_snapshots),
                'peak_memory_mb': max(s['memory_rss_mb'] for s in self.resource_snapshots),
                'avg_cpu_percent': sum(s['cpu_percent'] for s in self.resource_snapshots) / len(self.resource_snapshots),
                'snapshots': self.resource_snapshots
            }
        
        return summary


class LoggerSetup:
    """
    日志系统设置器
    
    职责：
    - 配置日志格式和级别
    - 支持同时输出到控制台和文件
    - 记录详细的时间戳和上下文信息
    - 提供分级日志功能
    """
    
    @staticmethod
    def setup_logger(
        name: str = "code_summarization",
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        log_dir: str = "logs"
    ) -> logging.Logger:
        """
        设置日志记录器
        
        实现要求 7.1: 记录所有关键操作的时间戳和状态信息
        实现要求 7.4: 同时输出到控制台和日志文件
        
        Args:
            name: 日志记录器名称
            log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: 日志文件名，如果为None则自动生成
            log_dir: 日志目录
            
        Returns:
            配置好的日志记录器
        """
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 如果没有指定日志文件名，自动生成（包含时间戳）
        if log_file is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"experiment_{timestamp}.log"
        
        log_path = os.path.join(log_dir, log_file)
        
        # 创建日志记录器
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)  # 设置为DEBUG以捕获所有级别
        
        # 清除已有的处理器（避免重复添加）
        logger.handlers.clear()
        
        # 创建详细的格式器（包含ISO 8601时间戳）
        detailed_formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s',
            datefmt='%Y-%m-%dT%H:%M:%S'  # ISO 8601格式
        )
        
        # 为控制台创建简洁的格式器
        console_formatter = logging.Formatter(
            fmt='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台处理器（使用用户指定的日志级别）
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # 文件处理器（记录所有DEBUG及以上级别）
        file_handler = logging.FileHandler(log_path, encoding='utf-8', mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
        # 防止日志传播到根记录器
        logger.propagate = False
        
        # 记录初始化信息
        logger.info("=" * 80)
        logger.info(f"Logger initialized: {name}")
        logger.info(f"Console log level: {log_level}")
        logger.info(f"File log level: DEBUG (all levels)")
        logger.info(f"Log file: {log_path}")
        logger.info(f"Timestamp format: ISO 8601")
        logger.info("=" * 80)
        
        return logger
    
    @staticmethod
    def log_experiment_start(logger: logging.Logger, config: dict) -> None:
        """
        记录实验开始信息
        
        Args:
            logger: 日志记录器
            config: 实验配置
        """
        logger.info("=" * 60)
        logger.info("EXPERIMENT STARTED")
        logger.info("=" * 60)
        logger.info(f"Experiment name: {config.get('experiment', {}).get('name', 'Unknown')}")
        logger.info(f"Start time: {datetime.datetime.now().isoformat()}")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Working directory: {os.getcwd()}")
        
        # 记录关键配置
        model_config = config.get('model', {})
        data_config = config.get('data', {})
        
        logger.info("Key configurations:")
        logger.info(f"  Model: {model_config.get('name', 'Unknown')}")
        logger.info(f"  Learning rate: {model_config.get('learning_rate', 'Unknown')}")
        logger.info(f"  Batch size: {model_config.get('batch_size', 'Unknown')}")
        logger.info(f"  Epochs: {model_config.get('num_epochs', 'Unknown')}")
        logger.info(f"  Dataset: {data_config.get('dataset_name', 'Unknown')}")
        logger.info(f"  Language: {data_config.get('language', 'Unknown')}")
        logger.info("=" * 60)
    
    @staticmethod
    def log_experiment_end(logger: logging.Logger, start_time: datetime.datetime, success: bool = True) -> None:
        """
        记录实验结束信息
        
        Args:
            logger: 日志记录器
            start_time: 实验开始时间
            success: 实验是否成功完成
        """
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        
        logger.info("=" * 60)
        if success:
            logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
        else:
            logger.error("EXPERIMENT FAILED")
        logger.info("=" * 60)
        logger.info(f"End time: {end_time.isoformat()}")
        logger.info(f"Total duration: {duration}")
        logger.info(f"Duration (seconds): {duration.total_seconds():.2f}")
        logger.info("=" * 60)
    
    @staticmethod
    def log_error_with_context(
        logger: logging.Logger, 
        error: Exception, 
        context: str = "",
        additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        记录带上下文的错误信息
        
        实现要求 7.2: 记录完整的错误堆栈信息和上下文
        
        Args:
            logger: 日志记录器
            error: 异常对象
            context: 上下文信息
            additional_info: 额外的调试信息
        """
        logger.error("=" * 80)
        logger.error("ERROR OCCURRED")
        logger.error("=" * 80)
        
        # 记录错误类型和消息
        error_type = type(error).__name__
        error_msg = str(error)
        logger.error(f"Error Type: {error_type}")
        logger.error(f"Error Message: {error_msg}")
        
        # 记录上下文信息
        if context:
            logger.error(f"Context: {context}")
        
        # 记录额外信息
        if additional_info:
            logger.error("Additional Information:")
            for key, value in additional_info.items():
                logger.error(f"  {key}: {value}")
        
        # 记录完整的堆栈跟踪
        logger.error("Full Stack Trace:")
        logger.error("-" * 80)
        tb_lines = traceback.format_exception(type(error), error, error.__traceback__)
        for line in tb_lines:
            # 逐行记录堆栈信息，保持格式
            for subline in line.rstrip().split('\n'):
                logger.error(subline)
        logger.error("-" * 80)
        
        # 记录时间戳
        logger.error(f"Error Timestamp: {datetime.datetime.now().isoformat()}")
        logger.error("=" * 80)
    
    @staticmethod
    def log_progress(logger: logging.Logger, current: int, total: int, prefix: str = "Progress") -> None:
        """
        记录进度信息
        
        Args:
            logger: 日志记录器
            current: 当前进度
            total: 总数
            prefix: 前缀信息
        """
        percentage = (current / total) * 100 if total > 0 else 0
        logger.info(f"{prefix}: {current}/{total} ({percentage:.1f}%)")
    
    @staticmethod
    def generate_experiment_summary(
        logger: logging.Logger,
        metrics: ExperimentMetrics,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        生成实验摘要报告
        
        实现要求 7.5: 生成包含总耗时、最终指标、资源使用情况的摘要
        
        Args:
            logger: 日志记录器
            metrics: 实验指标记录器
            output_path: 摘要报告保存路径（可选）
            
        Returns:
            实验摘要字典
        """
        # 完成指标记录
        metrics.finalize()
        
        # 获取摘要数据
        summary = metrics.get_summary()
        
        # 记录到日志
        logger.info("=" * 80)
        logger.info("EXPERIMENT SUMMARY REPORT")
        logger.info("=" * 80)
        
        # 实验基本信息
        exp_info = summary['experiment_info']
        logger.info("Experiment Information:")
        logger.info(f"  Start Time: {exp_info['start_time']}")
        logger.info(f"  End Time: {exp_info['end_time']}")
        logger.info(f"  Total Duration: {exp_info['duration_formatted']}")
        logger.info(f"  Duration (seconds): {exp_info['duration_seconds']:.2f}")
        
        # 操作统计
        logger.info("")
        logger.info("Operations Summary:")
        logger.info(f"  Total Operations: {summary['operations_count']}")
        logger.info(f"  Total Errors: {summary['errors_count']}")
        
        # 关键指标
        if summary['metrics']:
            logger.info("")
            logger.info("Key Metrics:")
            for metric_name, metric_value in summary['metrics'].items():
                if isinstance(metric_value, float):
                    logger.info(f"  {metric_name}: {metric_value:.4f}")
                else:
                    logger.info(f"  {metric_name}: {metric_value}")
        
        # 资源使用情况
        if 'resource_usage' in summary:
            resource = summary['resource_usage']
            logger.info("")
            logger.info("Resource Usage:")
            logger.info(f"  Peak Memory (MB): {resource['peak_memory_mb']:.2f}")
            logger.info(f"  Average CPU (%): {resource['avg_cpu_percent']:.2f}")
            logger.info(f"  Resource Snapshots: {resource['snapshots_count']}")
            
            # 如果有GPU信息
            if resource['snapshots'] and 'gpu_memory_allocated_mb' in resource['snapshots'][-1]:
                final_snapshot = resource['snapshots'][-1]
                logger.info(f"  Final GPU Memory Allocated (MB): {final_snapshot['gpu_memory_allocated_mb']:.2f}")
                logger.info(f"  Final GPU Memory Reserved (MB): {final_snapshot['gpu_memory_reserved_mb']:.2f}")
        
        # 错误摘要
        if summary['errors']:
            logger.info("")
            logger.info("Errors Summary:")
            for i, error in enumerate(summary['errors'][:5], 1):  # 只显示前5个错误
                logger.info(f"  Error {i}: {error['error_type']} - {error['error_message'][:100]}")
            if len(summary['errors']) > 5:
                logger.info(f"  ... and {len(summary['errors']) - 5} more errors")
        
        logger.info("=" * 80)
        
        # 保存到文件
        if output_path:
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                logger.info(f"Experiment summary saved to: {output_path}")
            except Exception as e:
                logger.error(f"Failed to save experiment summary: {e}")
        
        return summary


def get_logger(name: str = "code_summarization") -> logging.Logger:
    """
    获取日志记录器的便捷函数
    
    Args:
        name: 日志记录器名称
        
    Returns:
        日志记录器
    """
    return logging.getLogger(name)


def create_experiment_metrics() -> ExperimentMetrics:
    """
    创建实验指标记录器的便捷函数
    
    Returns:
        实验指标记录器
    """
    return ExperimentMetrics()
