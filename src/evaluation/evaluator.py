"""
评估引擎模块

负责评估模型性能并计算指标
"""

import os
import json
import logging
from typing import Dict, Any, List, Tuple, Optional
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import Dataset
from tqdm import tqdm

try:
    from src.config.manager import ConfigManager
except ImportError:
    from config.manager import ConfigManager


class EvaluationError(Exception):
    """评估相关错误"""
    pass


class EvaluationEngine:
    """
    评估引擎
    
    职责：
    - 在测试集上生成代码摘要
    - 计算评估指标（BLEU-4、METEOR、ROUGE-L）
    - 按代码长度分组评估
    - 保存评估结果
    """
    
    def __init__(
        self,
        config: ConfigManager,
        model: AutoModelForSeq2SeqLM,
        tokenizer: AutoTokenizer
    ):
        """
        初始化评估引擎
        
        Args:
            config: 配置管理器实例
            model: 训练好的模型
            tokenizer: 对应的tokenizer
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(__name__)
        
        # 设置设备
        self.device = next(model.parameters()).device
        self.logger.info(f"Evaluation device: {self.device}")
        
        # 初始化评估指标计算器
        self._setup_metrics()
    
    def _setup_metrics(self) -> None:
        """设置评估指标计算器"""
        try:
            # 导入NLTK相关模块
            import nltk
            
            # 下载必要的NLTK数据（如果尚未下载）
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                self.logger.info("Downloading NLTK punkt tokenizer...")
                nltk.download('punkt', quiet=True)
            
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                self.logger.info("Downloading NLTK wordnet...")
                nltk.download('wordnet', quiet=True)
            
            try:
                nltk.data.find('corpora/omw-1.4')
            except LookupError:
                self.logger.info("Downloading NLTK omw-1.4...")
                nltk.download('omw-1.4', quiet=True)
            
            self.logger.info("Evaluation metrics setup completed")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup some NLTK data: {e}")
            self.logger.warning("Some metrics may not work properly")
    
    def generate_summaries(
        self,
        test_dataset: Dataset,
        batch_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        在测试集上生成代码摘要
        
        Args:
            test_dataset: 测试数据集
            batch_size: 批次大小（可选，默认使用配置中的值）
            
        Returns:
            List[Dict[str, Any]]: 生成结果列表，每个元素包含：
                - code: 原始代码
                - reference: 参考摘要
                - generated: 生成的摘要
                - code_length: 代码长度
        """
        if batch_size is None:
            batch_size = self.config.get("model.batch_size", 16)
        
        # 获取生成参数
        beam_size = self.config.get("evaluation.beam_size", 5)
        num_beams = self.config.get("evaluation.num_beams", 5)
        max_target_length = self.config.get("data.max_target_length", 128)
        
        self.logger.info("Starting summary generation...")
        self.logger.info(f"Test dataset size: {len(test_dataset)}")
        self.logger.info(f"Batch size: {batch_size}")
        self.logger.info(f"Beam size: {beam_size}")
        self.logger.info(f"Num beams: {num_beams}")
        
        # 将模型设置为评估模式
        self.model.eval()
        
        results = []
        
        # 批量生成
        with torch.no_grad():
            for i in tqdm(range(0, len(test_dataset), batch_size), desc="Generating summaries"):
                batch_end = min(i + batch_size, len(test_dataset))
                batch = test_dataset[i:batch_end]
                
                # 准备输入
                input_ids = torch.tensor(batch['input_ids']).to(self.device)
                attention_mask = torch.tensor(batch['attention_mask']).to(self.device)
                
                # 生成摘要（使用beam search）
                try:
                    generated_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=max_target_length,
                        num_beams=num_beams,
                        early_stopping=True,
                        no_repeat_ngram_size=2,
                    )
                    
                    # 解码生成的摘要
                    generated_summaries = self.tokenizer.batch_decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    
                    # 收集结果
                    for j in range(len(generated_summaries)):
                        code = batch['code'][j] if 'code' in batch else ""
                        reference = batch['summary'][j] if 'summary' in batch else ""
                        generated = generated_summaries[j]
                        
                        results.append({
                            'code': code,
                            'reference': reference,
                            'generated': generated,
                            'code_length': len(code)
                        })
                
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        self.logger.warning(f"CUDA OOM at batch {i}, skipping batch")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise
        
        self.logger.info(f"Generated {len(results)} summaries")
        
        # 验证生成的摘要非空
        empty_count = sum(1 for r in results if not r['generated'].strip())
        if empty_count > 0:
            self.logger.warning(f"Warning: {empty_count} empty summaries generated")
        
        return results
    
    def compute_metrics(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        计算评估指标
        
        Args:
            predictions: 预测的摘要列表
            references: 参考摘要列表
            
        Returns:
            Dict[str, float]: 包含各种评估指标的字典
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"Predictions and references must have same length: "
                f"{len(predictions)} vs {len(references)}"
            )
        
        self.logger.info("Computing evaluation metrics...")
        
        metrics = {}
        
        # 计算BLEU-4
        try:
            bleu_score = self._compute_bleu(predictions, references)
            metrics['bleu_4'] = bleu_score
            self.logger.info(f"BLEU-4: {bleu_score:.4f}")
        except Exception as e:
            self.logger.error(f"Failed to compute BLEU: {e}")
            metrics['bleu_4'] = 0.0
        
        # 计算METEOR
        try:
            meteor_score = self._compute_meteor(predictions, references)
            metrics['meteor'] = meteor_score
            self.logger.info(f"METEOR: {meteor_score:.4f}")
        except Exception as e:
            self.logger.error(f"Failed to compute METEOR: {e}")
            metrics['meteor'] = 0.0
        
        # 计算ROUGE-L
        try:
            rouge_scores = self._compute_rouge(predictions, references)
            metrics['rouge_l'] = rouge_scores['rouge_l']
            self.logger.info(f"ROUGE-L: {rouge_scores['rouge_l']:.4f}")
        except Exception as e:
            self.logger.error(f"Failed to compute ROUGE: {e}")
            metrics['rouge_l'] = 0.0
        
        return metrics
    
    def _compute_bleu(
        self,
        predictions: List[str],
        references: List[str]
    ) -> float:
        """
        计算BLEU-4指标
        
        Args:
            predictions: 预测的摘要列表
            references: 参考摘要列表
            
        Returns:
            float: BLEU-4分数
        """
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        smooth = SmoothingFunction()
        scores = []
        
        for pred, ref in zip(predictions, references):
            # 分词
            pred_tokens = pred.split()
            ref_tokens = ref.split()
            
            # 计算BLEU-4（使用平滑函数避免0分）
            score = sentence_bleu(
                [ref_tokens],
                pred_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smooth.method1
            )
            scores.append(score)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _compute_meteor(
        self,
        predictions: List[str],
        references: List[str]
    ) -> float:
        """
        计算METEOR指标
        
        Args:
            predictions: 预测的摘要列表
            references: 参考摘要列表
            
        Returns:
            float: METEOR分数
        """
        from nltk.translate.meteor_score import meteor_score
        
        scores = []
        
        for pred, ref in zip(predictions, references):
            # 分词
            pred_tokens = pred.split()
            ref_tokens = ref.split()
            
            # 计算METEOR
            score = meteor_score([ref_tokens], pred_tokens)
            scores.append(score)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _compute_rouge(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        计算ROUGE-L指标
        
        Args:
            predictions: 预测的摘要列表
            references: 参考摘要列表
            
        Returns:
            Dict[str, float]: ROUGE分数字典
        """
        from rouge_score import rouge_scorer
        
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = []
        
        for pred, ref in zip(predictions, references):
            score = scorer.score(ref, pred)
            scores.append(score['rougeL'].fmeasure)
        
        return {
            'rouge_l': sum(scores) / len(scores) if scores else 0.0
        }
    
    def evaluate_by_length(
        self,
        test_dataset: Dataset,
        length_bins: Optional[List[Tuple[int, int]]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        按代码长度区间对测试样本分组评估
        
        Args:
            test_dataset: 测试数据集
            length_bins: 长度区间列表，格式为 [(min1, max1), (min2, max2), ...]
                        如果为None，则使用默认区间
            
        Returns:
            Dict[str, Dict[str, float]]: 每个长度区间的评估指标
        """
        # 默认长度区间
        if length_bins is None:
            length_bins = [
                (0, 100),      # 短代码
                (100, 300),    # 中等代码
                (300, 1000)    # 长代码
            ]
        
        self.logger.info("Starting evaluation by code length...")
        self.logger.info(f"Length bins: {length_bins}")
        
        # 首先生成所有摘要
        all_results = self.generate_summaries(test_dataset)
        
        # 按长度分组
        grouped_results = {
            f"{min_len}-{max_len}": []
            for min_len, max_len in length_bins
        }
        
        for result in all_results:
            code_length = result['code_length']
            
            # 找到对应的区间
            for min_len, max_len in length_bins:
                if min_len <= code_length < max_len:
                    grouped_results[f"{min_len}-{max_len}"].append(result)
                    break
        
        # 计算每个组的指标
        metrics_by_length = {}
        
        for length_range, results in grouped_results.items():
            if not results:
                self.logger.warning(f"No samples in length range {length_range}")
                metrics_by_length[length_range] = {
                    'sample_count': 0,
                    'bleu_4': 0.0,
                    'meteor': 0.0,
                    'rouge_l': 0.0
                }
                continue
            
            predictions = [r['generated'] for r in results]
            references = [r['reference'] for r in results]
            
            metrics = self.compute_metrics(predictions, references)
            metrics['sample_count'] = len(results)
            
            metrics_by_length[length_range] = metrics
            
            self.logger.info(f"Length range {length_range}: {len(results)} samples")
            self.logger.info(f"  BLEU-4: {metrics['bleu_4']:.4f}")
            self.logger.info(f"  METEOR: {metrics['meteor']:.4f}")
            self.logger.info(f"  ROUGE-L: {metrics['rouge_l']:.4f}")
        
        # 验证分组评估一致性
        total_samples = sum(m['sample_count'] for m in metrics_by_length.values())
        if total_samples != len(all_results):
            self.logger.warning(
                f"Sample count mismatch: {total_samples} vs {len(all_results)}"
            )
        
        return metrics_by_length
    
    def save_results(
        self,
        results: Dict[str, Any],
        output_path: str
    ) -> None:
        """
        保存评估结果
        
        Args:
            results: 评估结果字典
            output_path: 输出文件路径
        """
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 保存为JSON格式
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Evaluation results saved to: {output_path}")
            
        except Exception as e:
            error_msg = f"Failed to save results to {output_path}: {e}"
            self.logger.error(error_msg)
            raise EvaluationError(error_msg)
    
    def evaluate(
        self,
        test_dataset: Dataset,
        save_samples: bool = True,
        num_samples: int = 10
    ) -> Dict[str, Any]:
        """
        完整的评估流程
        
        Args:
            test_dataset: 测试数据集
            save_samples: 是否保存样例
            num_samples: 保存的样例数量
            
        Returns:
            Dict[str, Any]: 完整的评估结果
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting full evaluation...")
        self.logger.info("=" * 60)
        
        # 生成摘要
        generation_results = self.generate_summaries(test_dataset)
        
        # 提取预测和参考
        predictions = [r['generated'] for r in generation_results]
        references = [r['reference'] for r in generation_results]
        
        # 计算整体指标
        self.logger.info("\nOverall metrics:")
        overall_metrics = self.compute_metrics(predictions, references)
        
        # 按长度分组评估
        self.logger.info("\nMetrics by code length:")
        length_metrics = self.evaluate_by_length(test_dataset)
        
        # 准备完整结果
        full_results = {
            'model_name': self.config.get('model.name'),
            'dataset_size': len(test_dataset),
            'overall_metrics': overall_metrics,
            'metrics_by_length': length_metrics,
        }
        
        # 保存样例
        if save_samples and generation_results:
            import random
            sample_indices = random.sample(
                range(len(generation_results)),
                min(num_samples, len(generation_results))
            )
            samples = [generation_results[i] for i in sample_indices]
            full_results['sample_outputs'] = samples
            
            self.logger.info(f"\nSaved {len(samples)} sample outputs")
        
        self.logger.info("=" * 60)
        self.logger.info("Evaluation completed!")
        self.logger.info("=" * 60)
        
        return full_results
