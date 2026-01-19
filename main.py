#!/usr/bin/env python3
"""
Code Summarization Experiment - Main Entry Point
代码摘要生成实验 - 主程序入口

Usage:
    python main.py --config config/default.yaml
    python main.py --config config/default.yaml --model_name Salesforce/codet5-small
"""

import os
import sys
import argparse
import datetime
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config.manager import ConfigManager, ConfigError
from utils.logger import LoggerSetup, ExperimentMetrics, get_logger, create_experiment_metrics
from data.dataset_processor import DatasetProcessor
from models.trainer import ModelTrainer, ModelError


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Code Summarization Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --config config/default.yaml
    python main.py --config config/default.yaml --model_name microsoft/codebert-base
    python main.py --config config/default.yaml --learning_rate 1e-4 --batch_size 8
    python main.py --show_params  # Show all valid parameter ranges
        """
    )
    
    # 必需参数
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/default.yaml',
        help='Path to configuration file (default: config/default.yaml)'
    )
    
    # 显示参数信息
    parser.add_argument(
        '--show_params',
        action='store_true',
        help='Show all valid parameter ranges and examples'
    )
    
    # 可选的配置覆盖参数
    parser.add_argument('--model_name', type=str, help='Model name to use')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--num_epochs', type=int, help='Number of training epochs')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Log level')
    
    # 运行模式
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['full', 'data_only', 'train_only', 'eval_only'], 
        default='full',
        help='Execution mode (default: full)'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    start_time = datetime.datetime.now()
    logger = None
    experiment_metrics = None
    
    try:
        # 创建实验指标记录器
        experiment_metrics = create_experiment_metrics()
        
        # 解析命令行参数
        args = parse_arguments()
        experiment_metrics.record_operation("parse_arguments", "success")
        
        # 如果用户请求显示参数信息
        if args.show_params:
            # 创建一个临时配置管理器来获取参数信息
            try:
                temp_config = ConfigManager(args.config)
                print(temp_config.get_parameter_info())
            except Exception:
                # 如果配置文件不存在，仍然显示参数信息
                print(ConfigManager.__dict__['get_parameter_info'](None))
            return
        
        # 加载配置
        print(f"Loading configuration from: {args.config}")
        config_manager = ConfigManager(args.config)
        experiment_metrics.record_operation("load_config", "success", {"config_file": args.config})
        
        # 使用命令行参数覆盖配置
        cli_args = {
            'model_name': args.model_name,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'num_epochs': args.num_epochs,
            'output_dir': args.output_dir,
            'log_level': args.log_level,
        }
        config_manager.update_from_args(cli_args)
        experiment_metrics.record_operation("update_config_from_args", "success")
        
        # 设置日志系统
        log_level = config_manager.get('logging.log_level', 'INFO')
        log_file = config_manager.get('logging.log_file', 'experiment.log')
        logger = LoggerSetup.setup_logger(log_level=log_level, log_file=log_file)
        experiment_metrics.record_operation("setup_logger", "success", {"log_level": log_level, "log_file": log_file})
        
        # 记录实验开始
        LoggerSetup.log_experiment_start(logger, config_manager.to_dict())
        experiment_metrics.snapshot_resources()
        
        # 创建输出目录
        output_dir = config_manager.get('experiment.output_dir', './outputs')
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        experiment_metrics.record_operation("create_output_dir", "success", {"output_dir": output_dir})
        
        # 保存配置文件副本
        config_backup_path = os.path.join(output_dir, 'config_backup.yaml')
        config_manager.save_config(config_backup_path)
        logger.info(f"Configuration backup saved to: {config_backup_path}")
        experiment_metrics.record_operation("save_config_backup", "success")
        
        # 根据模式执行不同的任务
        logger.info(f"Execution mode: {args.mode}")
        experiment_metrics.record_metric("execution_mode", args.mode)
        
        if args.mode == 'full':
            logger.info("Running full experiment pipeline...")
            experiment_metrics.record_operation("start_full_pipeline", "started")
            
            # Step 1: Data Processing
            logger.info("=" * 60)
            logger.info("Step 1: Data Processing")
            logger.info("=" * 60)
            
            local_processed_path = config_manager.get("data.local_processed_data", "./data/processed")
            
            # Check if processed data already exists
            if os.path.exists(local_processed_path) and len(os.listdir(local_processed_path)) > 0:
                logger.info(f"Found existing processed data at: {local_processed_path}")
                logger.info("Loading processed data...")
                processor = DatasetProcessor(config_manager)
                train_dataset, val_dataset, test_dataset = processor.load_processed_data(local_processed_path)
                experiment_metrics.record_operation("load_processed_data", "success")
            else:
                logger.info("Processing data from scratch...")
                data_dir = local_processed_path
                os.makedirs(data_dir, exist_ok=True)
                
                processor = DatasetProcessor(config_manager)
                dataset = processor.download_dataset()
                stats = processor.get_data_statistics()
                experiment_metrics.record_metric("dataset_statistics", stats)
                
                model_name = config_manager.get("model.name", "Salesforce/codet5-small")
                processor.setup_tokenizer(model_name)
                
                train_dataset, val_dataset, test_dataset = processor.prepare_datasets()
                processor.save_processed_data(train_dataset, val_dataset, test_dataset, data_dir)
                
                if not processor.check_data_integrity(data_dir):
                    raise Exception("Data integrity check failed")
                
                experiment_metrics.record_operation("process_data", "success")
            
            experiment_metrics.snapshot_resources()
            logger.info(f"Data ready: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
            experiment_metrics.record_metric("train_size", len(train_dataset))
            experiment_metrics.record_metric("val_size", len(val_dataset))
            experiment_metrics.record_metric("test_size", len(test_dataset))
            
            # Step 2: Model Training
            logger.info("=" * 60)
            logger.info("Step 2: Model Training")
            logger.info("=" * 60)
            
            trainer = ModelTrainer(config_manager)
            model, tokenizer = trainer.load_model()
            experiment_metrics.record_operation("load_model", "success")
            experiment_metrics.snapshot_resources()
            
            trainer.setup_training(train_dataset, val_dataset)
            experiment_metrics.record_operation("setup_training", "success")
            
            history = trainer.train()
            experiment_metrics.record_operation("train_model", "success")
            experiment_metrics.snapshot_resources()
            
            model_save_path = os.path.join(output_dir, "best_model")
            trainer.save_checkpoint(model_save_path)
            experiment_metrics.record_operation("save_model", "success")
            
            history_path = os.path.join(output_dir, "training_history.json")
            with open(history_path, 'w', encoding='utf-8') as f:
                import json
                json.dump(history, f, indent=2)
            
            # Record final metrics
            if 'eval_loss' in history:
                experiment_metrics.record_metric("final_eval_loss", history['eval_loss'][-1] if history['eval_loss'] else None)
            
            logger.info("=" * 60)
            logger.info("Full pipeline completed successfully!")
            logger.info("=" * 60)
            logger.info(f"Model saved to: {model_save_path}")
            logger.info(f"Training history saved to: {history_path}")
            experiment_metrics.record_operation("full_pipeline", "completed")
            
        elif args.mode == 'data_only':
            logger.info("Running data processing only...")
            experiment_metrics.record_operation("start_data_only", "started")
            
            # 检查是否已有处理后的数据
            local_processed_path = config_manager.get("data.local_processed_data", "./data/processed")
            if os.path.exists(local_processed_path) and len(os.listdir(local_processed_path)) > 0:
                logger.info(f"发现已处理的数据: {local_processed_path}")
                logger.info("如需重新处理，请删除该目录或使用不同的输出路径")
            
            # 创建数据输出目录
            data_dir = local_processed_path
            os.makedirs(data_dir, exist_ok=True)
            
            # 创建数据处理器
            logger.info("Initializing dataset processor...")
            processor = DatasetProcessor(config_manager)
            experiment_metrics.record_operation("init_processor", "success")
            
            # 下载数据集
            logger.info("Downloading CodeSearchNet dataset...")
            dataset = processor.download_dataset()
            experiment_metrics.record_operation("download_dataset", "success")
            experiment_metrics.snapshot_resources()
            
            # 获取统计信息
            logger.info("Computing dataset statistics...")
            stats = processor.get_data_statistics()
            experiment_metrics.record_metric("dataset_statistics", stats)
            experiment_metrics.record_operation("compute_statistics", "success")
            
            # 设置tokenizer
            model_name = config_manager.get("model.name", "Salesforce/codet5-small")
            logger.info(f"Setting up tokenizer: {model_name}")
            processor.setup_tokenizer(model_name)
            experiment_metrics.record_operation("setup_tokenizer", "success")
            
            # 预处理数据集
            logger.info("Preprocessing datasets...")
            train_dataset, val_dataset, test_dataset = processor.prepare_datasets()
            logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
            experiment_metrics.record_metric("train_size", len(train_dataset))
            experiment_metrics.record_metric("val_size", len(val_dataset))
            experiment_metrics.record_metric("test_size", len(test_dataset))
            experiment_metrics.record_operation("prepare_datasets", "success")
            experiment_metrics.snapshot_resources()
            
            # 保存处理后的数据
            logger.info("Saving processed data...")
            processor.save_processed_data(train_dataset, val_dataset, test_dataset, data_dir)
            experiment_metrics.record_operation("save_processed_data", "success")
            
            # 验证数据完整性
            logger.info("Verifying data integrity...")
            if processor.check_data_integrity(data_dir):
                logger.info("Data integrity check passed")
                experiment_metrics.record_operation("verify_integrity", "success")
            else:
                logger.error("Data integrity check failed")
                experiment_metrics.record_operation("verify_integrity", "failed")
                experiment_metrics.record_error("DataIntegrityError", "Data integrity check failed", "data_only mode")
                raise Exception("Data integrity check failed")
            
            logger.info(f"Data processing completed. Output: {data_dir}")
            experiment_metrics.record_operation("data_only", "completed")
            
        elif args.mode == 'train_only':
            logger.info("Running training only...")
            experiment_metrics.record_operation("start_train_only", "started")
            
            # 检查是否有处理后的数据
            local_processed_path = config_manager.get("data.local_processed_data", "./data/processed")
            if not os.path.exists(local_processed_path):
                logger.error(f"Processed data not found at: {local_processed_path}")
                logger.error("Please run with --mode data_only first to process the data")
                experiment_metrics.record_error("DataNotFoundError", f"Processed data not found at: {local_processed_path}", "train_only mode")
                raise Exception("Processed data not found")
            
            # 创建数据处理器并加载数据
            logger.info("Loading processed data...")
            processor = DatasetProcessor(config_manager)
            train_dataset, val_dataset, test_dataset = processor.load_processed_data(local_processed_path)
            experiment_metrics.record_operation("load_processed_data", "success")
            experiment_metrics.record_metric("train_size", len(train_dataset))
            experiment_metrics.record_metric("val_size", len(val_dataset))
            experiment_metrics.record_metric("test_size", len(test_dataset))
            experiment_metrics.snapshot_resources()
            
            # 创建模型训练器
            logger.info("Initializing model trainer...")
            trainer = ModelTrainer(config_manager)
            experiment_metrics.record_operation("init_trainer", "success")
            
            # 加载模型
            logger.info("Loading model...")
            model, tokenizer = trainer.load_model()
            experiment_metrics.record_operation("load_model", "success")
            experiment_metrics.snapshot_resources()
            
            # 设置训练
            logger.info("Setting up training...")
            trainer.setup_training(train_dataset, val_dataset)
            experiment_metrics.record_operation("setup_training", "success")
            
            # 执行训练
            logger.info("Starting training...")
            history = trainer.train()
            experiment_metrics.record_operation("train_model", "success")
            experiment_metrics.snapshot_resources()
            
            # Record final metrics
            if 'eval_loss' in history:
                experiment_metrics.record_metric("final_eval_loss", history['eval_loss'][-1] if history['eval_loss'] else None)
            
            # 保存模型
            model_save_path = os.path.join(output_dir, "best_model")
            logger.info(f"Saving model to: {model_save_path}")
            trainer.save_checkpoint(model_save_path)
            experiment_metrics.record_operation("save_model", "success")
            
            # 保存训练历史
            history_path = os.path.join(output_dir, "training_history.json")
            with open(history_path, 'w', encoding='utf-8') as f:
                import json
                json.dump(history, f, indent=2)
            logger.info(f"Training history saved to: {history_path}")
            
            logger.info("Training completed successfully!")
            experiment_metrics.record_operation("train_only", "completed")
            
        elif args.mode == 'eval_only':
            logger.info("Running evaluation only...")
            experiment_metrics.record_operation("start_eval_only", "started")
            
            # 检查是否有处理后的数据
            local_processed_path = config_manager.get("data.local_processed_data", "./data/processed")
            if not os.path.exists(local_processed_path):
                logger.error(f"Processed data not found at: {local_processed_path}")
                logger.error("Please run with --mode data_only first to process the data")
                experiment_metrics.record_error("DataNotFoundError", f"Processed data not found at: {local_processed_path}", "eval_only mode")
                raise Exception("Processed data not found")
            
            # 检查是否有训练好的模型
            model_path = os.path.join(output_dir, "best_model")
            if not os.path.exists(model_path):
                logger.error(f"Trained model not found at: {model_path}")
                logger.error("Please run with --mode train_only first to train the model")
                experiment_metrics.record_error("ModelNotFoundError", f"Trained model not found at: {model_path}", "eval_only mode")
                raise Exception("Trained model not found")
            
            # 加载处理后的数据
            logger.info("Loading processed data...")
            processor = DatasetProcessor(config_manager)
            train_dataset, val_dataset, test_dataset = processor.load_processed_data(local_processed_path)
            experiment_metrics.record_operation("load_processed_data", "success")
            experiment_metrics.record_metric("test_size", len(test_dataset))
            experiment_metrics.snapshot_resources()
            
            # 加载训练好的模型
            logger.info(f"Loading trained model from: {model_path}")
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            experiment_metrics.record_operation("load_trained_model", "success")
            experiment_metrics.snapshot_resources()
            
            # 创建评估引擎
            logger.info("Initializing evaluation engine...")
            from evaluation.evaluator import EvaluationEngine
            evaluator = EvaluationEngine(config_manager, model, tokenizer)
            experiment_metrics.record_operation("init_evaluator", "success")
            
            # 执行评估
            logger.info("Starting evaluation...")
            eval_results = evaluator.evaluate(test_dataset, save_samples=True, num_samples=10)
            experiment_metrics.record_operation("evaluate_model", "success")
            experiment_metrics.snapshot_resources()
            
            # 记录评估指标
            if 'overall_metrics' in eval_results:
                for metric_name, metric_value in eval_results['overall_metrics'].items():
                    experiment_metrics.record_metric(f"eval_{metric_name}", metric_value)
            
            # 保存评估结果
            eval_results_path = os.path.join(output_dir, "evaluation_results.json")
            evaluator.save_results(eval_results, eval_results_path)
            logger.info(f"Evaluation results saved to: {eval_results_path}")
            experiment_metrics.record_operation("save_eval_results", "success")
            
            # 生成可视化报告
            logger.info("Generating visualization reports...")
            from visualization.report_generator import ReportGenerator
            report_gen = ReportGenerator(config_manager)
            
            # 加载训练历史（如果存在）
            history_path = os.path.join(output_dir, "training_history.json")
            training_history = None
            if os.path.exists(history_path):
                with open(history_path, 'r', encoding='utf-8') as f:
                    import json
                    training_history = json.load(f)
            
            # 生成完整报告
            report_gen.generate_full_report(eval_results, training_history)
            experiment_metrics.record_operation("generate_report", "success")
            
            logger.info("Evaluation completed successfully!")
            experiment_metrics.record_operation("eval_only", "completed")
        
        # 生成实验摘要报告
        summary_path = os.path.join(output_dir, "experiment_summary.json")
        LoggerSetup.generate_experiment_summary(logger, experiment_metrics, summary_path)
        
        # 记录实验成功完成
        LoggerSetup.log_experiment_end(logger, start_time, success=True)
        print(f"\nExperiment completed successfully!")
        print(f"Check logs at: logs/{log_file}")
        print(f"Check outputs at: {output_dir}")
        print(f"Experiment summary: {summary_path}")
        
    except ConfigError as e:
        error_msg = f"Configuration error: {e}"
        if logger:
            LoggerSetup.log_error_with_context(logger, e, "Configuration validation")
            if experiment_metrics:
                experiment_metrics.record_error("ConfigError", str(e), "Configuration validation")
                summary_path = os.path.join(output_dir if 'output_dir' in locals() else './outputs', "experiment_summary.json")
                LoggerSetup.generate_experiment_summary(logger, experiment_metrics, summary_path)
        else:
            print(f"ERROR: {error_msg}", file=sys.stderr)
            print("\nFor valid parameter ranges, run: python main.py --show_params", file=sys.stderr)
        sys.exit(1)
        
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        if logger:
            LoggerSetup.log_error_with_context(logger, e, "Main execution")
            if experiment_metrics:
                experiment_metrics.record_error(type(e).__name__, str(e), "Main execution")
                summary_path = os.path.join(output_dir if 'output_dir' in locals() else './outputs', "experiment_summary.json")
                LoggerSetup.generate_experiment_summary(logger, experiment_metrics, summary_path)
            LoggerSetup.log_experiment_end(logger, start_time, success=False)
        else:
            print(f"ERROR: {error_msg}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()