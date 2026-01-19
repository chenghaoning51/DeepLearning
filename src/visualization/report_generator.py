"""
报告生成器模块

负责生成实验报告和可视化
"""

import os
import json
import logging
import random
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

try:
    from src.config.manager import ConfigManager
except ImportError:
    from config.manager import ConfigManager


class ReportError(Exception):
    """报告生成相关错误"""
    pass


class ReportGenerator:
    """
    报告生成器
    
    职责：
    - 创建模型性能对比表格
    - 生成训练过程可视化图表
    - 分析代码长度对性能的影响
    - 展示生成样例
    - 保存所有图表和报告
    """
    
    def __init__(self, config: ConfigManager):
        """
        初始化报告生成器
        
        Args:
            config: 配置管理器实例
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 设置matplotlib样式
        self._setup_plot_style()
        
        self.logger.info("ReportGenerator initialized")
    
    def _setup_plot_style(self) -> None:
        """设置绘图样式"""
        # 使用seaborn样式
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.2)
        
        # 设置matplotlib参数
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        
        self.logger.info("Plot style configured")
    
    def create_performance_table(
        self,
        results: Dict[str, Any],
        output_formats: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        创建模型性能对比表格
        
        Args:
            results: 评估结果字典，包含overall_metrics和metrics_by_length
            output_formats: 输出格式列表，可选值：['csv', 'latex', 'markdown']
                          如果为None，则使用所有格式
            
        Returns:
            pd.DataFrame: 性能对比表格
        """
        if output_formats is None:
            output_formats = ['csv', 'latex', 'markdown']
        
        self.logger.info("Creating performance comparison table...")
        
        # 提取整体指标
        overall_metrics = results.get('overall_metrics', {})
        metrics_by_length = results.get('metrics_by_length', {})
        model_name = results.get('model_name', 'Unknown')
        dataset_size = results.get('dataset_size', 0)
        
        # 创建表格数据
        table_data = []
        
        # 添加整体性能行
        table_data.append({
            'Model': model_name,
            'Dataset Size': dataset_size,
            'Code Length': 'Overall',
            'Sample Count': dataset_size,
            'BLEU-4': overall_metrics.get('bleu_4', 0.0),
            'METEOR': overall_metrics.get('meteor', 0.0),
            'ROUGE-L': overall_metrics.get('rouge_l', 0.0)
        })
        
        # 添加按长度分组的性能行
        for length_range, metrics in sorted(metrics_by_length.items()):
            table_data.append({
                'Model': model_name,
                'Dataset Size': dataset_size,
                'Code Length': length_range,
                'Sample Count': metrics.get('sample_count', 0),
                'BLEU-4': metrics.get('bleu_4', 0.0),
                'METEOR': metrics.get('meteor', 0.0),
                'ROUGE-L': metrics.get('rouge_l', 0.0)
            })
        
        # 创建DataFrame
        df = pd.DataFrame(table_data)
        
        # 格式化数值列
        for col in ['BLEU-4', 'METEOR', 'ROUGE-L']:
            df[col] = df[col].apply(lambda x: f"{x:.4f}")
        
        self.logger.info(f"Performance table created with {len(df)} rows")
        
        # 保存为不同格式
        output_dir = self.config.get("experiment.output_dir", "./outputs")
        
        if 'csv' in output_formats:
            csv_path = os.path.join(output_dir, "performance_table.csv")
            df.to_csv(csv_path, index=False, encoding='utf-8')
            self.logger.info(f"CSV table saved to: {csv_path}")
        
        if 'latex' in output_formats:
            latex_path = os.path.join(output_dir, "performance_table.tex")
            latex_str = df.to_latex(index=False, float_format="%.4f")
            with open(latex_path, 'w', encoding='utf-8') as f:
                f.write(latex_str)
            self.logger.info(f"LaTeX table saved to: {latex_path}")
        
        if 'markdown' in output_formats:
            markdown_path = os.path.join(output_dir, "performance_table.md")
            markdown_str = df.to_markdown(index=False)
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(markdown_str)
            self.logger.info(f"Markdown table saved to: {markdown_path}")
        
        return df
    
    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        save_path: Optional[str] = None
    ) -> None:
        """
        生成训练/验证损失曲线图
        
        Args:
            history: 训练历史字典，包含train_loss, eval_loss, learning_rate等
            save_path: 保存路径，如果为None则使用默认路径
        """
        self.logger.info("Plotting training curves...")
        
        # 如果没有指定保存路径，使用默认路径
        if save_path is None:
            output_dir = self.config.get("experiment.output_dir", "./outputs")
            save_path = os.path.join(output_dir, "training_curves.png")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 创建图表
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 绘制训练和验证损失
        ax1 = axes[0]
        if 'train_loss' in history and history['train_loss']:
            train_loss = history['train_loss']
            ax1.plot(train_loss, label='Training Loss', linewidth=2, color='#2E86AB')
        
        if 'eval_loss' in history and history['eval_loss']:
            eval_loss = history['eval_loss']
            # 验证损失通常是每个epoch记录一次，需要调整x轴
            if 'train_loss' in history and history['train_loss']:
                # 计算每个epoch对应的步数
                steps_per_epoch = len(history['train_loss']) // len(eval_loss) if eval_loss else 1
                eval_steps = [i * steps_per_epoch for i in range(len(eval_loss))]
                ax1.plot(eval_steps, eval_loss, label='Validation Loss', 
                        linewidth=2, color='#A23B72', marker='o', markersize=6)
            else:
                ax1.plot(eval_loss, label='Validation Loss', 
                        linewidth=2, color='#A23B72', marker='o', markersize=6)
        
        ax1.set_xlabel('Training Steps', fontsize=11)
        ax1.set_ylabel('Loss', fontsize=11)
        ax1.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', frameon=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        
        # 绘制学习率变化
        ax2 = axes[1]
        if 'learning_rate' in history and history['learning_rate']:
            learning_rate = history['learning_rate']
            ax2.plot(learning_rate, linewidth=2, color='#F18F01')
            ax2.set_xlabel('Training Steps', fontsize=11)
            ax2.set_ylabel('Learning Rate', fontsize=11)
            ax2.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # 使用科学计数法显示y轴
            ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
        else:
            ax2.text(0.5, 0.5, 'No learning rate data available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training curves saved to: {save_path}")
    
    def create_learning_rate_plot(
        self,
        history: Dict[str, List[float]],
        save_path: Optional[str] = None
    ) -> None:
        """
        创建学习率变化图表（单独的详细图表）
        
        Args:
            history: 训练历史字典
            save_path: 保存路径
        """
        self.logger.info("Creating learning rate plot...")
        
        if 'learning_rate' not in history or not history['learning_rate']:
            self.logger.warning("No learning rate data available")
            return
        
        # 如果没有指定保存路径，使用默认路径
        if save_path is None:
            output_dir = self.config.get("experiment.output_dir", "./outputs")
            save_path = os.path.join(output_dir, "learning_rate.png")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 6))
        
        learning_rate = history['learning_rate']
        steps = list(range(len(learning_rate)))
        
        ax.plot(steps, learning_rate, linewidth=2, color='#F18F01')
        ax.set_xlabel('Training Steps', fontsize=11)
        ax.set_ylabel('Learning Rate', fontsize=11)
        ax.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 使用科学计数法显示y轴
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
        
        # 添加统计信息
        max_lr = max(learning_rate)
        min_lr = min(learning_rate)
        final_lr = learning_rate[-1]
        
        info_text = f"Max LR: {max_lr:.2e}\nMin LR: {min_lr:.2e}\nFinal LR: {final_lr:.2e}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor='wheat', alpha=0.5), fontsize=9)
        
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Learning rate plot saved to: {save_path}")
    
    def plot_length_analysis(
        self,
        length_results: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None
    ) -> None:
        """
        生成不同代码长度区间的性能对比图
        
        Args:
            length_results: 按长度分组的评估结果
            save_path: 保存路径
        """
        self.logger.info("Plotting code length analysis...")
        
        # 如果没有指定保存路径，使用默认路径
        if save_path is None:
            output_dir = self.config.get("experiment.output_dir", "./outputs")
            save_path = os.path.join(output_dir, "length_analysis.png")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 准备数据
        length_ranges = []
        bleu_scores = []
        meteor_scores = []
        rouge_scores = []
        sample_counts = []
        
        for length_range, metrics in sorted(length_results.items()):
            length_ranges.append(length_range)
            bleu_scores.append(metrics.get('bleu_4', 0.0))
            meteor_scores.append(metrics.get('meteor', 0.0))
            rouge_scores.append(metrics.get('rouge_l', 0.0))
            sample_counts.append(metrics.get('sample_count', 0))
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 所有指标对比（条形图）
        ax1 = axes[0, 0]
        x = np.arange(len(length_ranges))
        width = 0.25
        
        ax1.bar(x - width, bleu_scores, width, label='BLEU-4', color='#2E86AB', alpha=0.8)
        ax1.bar(x, meteor_scores, width, label='METEOR', color='#A23B72', alpha=0.8)
        ax1.bar(x + width, rouge_scores, width, label='ROUGE-L', color='#F18F01', alpha=0.8)
        
        ax1.set_xlabel('Code Length Range', fontsize=11)
        ax1.set_ylabel('Score', fontsize=11)
        ax1.set_title('Performance Metrics by Code Length', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(length_ranges, rotation=45, ha='right')
        ax1.legend(loc='best', frameon=True, shadow=True)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, max(max(bleu_scores), max(meteor_scores), max(rouge_scores)) * 1.1)
        
        # 2. BLEU-4趋势线
        ax2 = axes[0, 1]
        x_pos = np.arange(len(length_ranges))
        ax2.plot(x_pos, bleu_scores, marker='o', linewidth=2, 
                markersize=8, color='#2E86AB', label='BLEU-4')
        ax2.set_xlabel('Code Length Range', fontsize=11)
        ax2.set_ylabel('BLEU-4 Score', fontsize=11)
        ax2.set_title('BLEU-4 Score vs Code Length', fontsize=12, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(length_ranges, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', frameon=True, shadow=True)
        
        # 3. METEOR和ROUGE-L趋势线
        ax3 = axes[1, 0]
        ax3.plot(x_pos, meteor_scores, marker='s', linewidth=2, 
                markersize=8, color='#A23B72', label='METEOR')
        ax3.plot(x_pos, rouge_scores, marker='^', linewidth=2, 
                markersize=8, color='#F18F01', label='ROUGE-L')
        ax3.set_xlabel('Code Length Range', fontsize=11)
        ax3.set_ylabel('Score', fontsize=11)
        ax3.set_title('METEOR & ROUGE-L vs Code Length', fontsize=12, fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(length_ranges, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='best', frameon=True, shadow=True)
        
        # 4. 样本数量分布
        ax4 = axes[1, 1]
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(length_ranges)))
        bars = ax4.bar(x_pos, sample_counts, color=colors, alpha=0.8)
        ax4.set_xlabel('Code Length Range', fontsize=11)
        ax4.set_ylabel('Sample Count', fontsize=11)
        ax4.set_title('Sample Distribution by Code Length', fontsize=12, fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(length_ranges, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 在条形图上添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Length analysis plot saved to: {save_path}")
    
    def create_length_performance_heatmap(
        self,
        length_results: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None
    ) -> None:
        """
        创建长度-性能关系的热力图
        
        Args:
            length_results: 按长度分组的评估结果
            save_path: 保存路径
        """
        self.logger.info("Creating length-performance heatmap...")
        
        # 如果没有指定保存路径，使用默认路径
        if save_path is None:
            output_dir = self.config.get("experiment.output_dir", "./outputs")
            save_path = os.path.join(output_dir, "length_performance_heatmap.png")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 准备数据
        length_ranges = []
        metrics_data = []
        
        for length_range, metrics in sorted(length_results.items()):
            length_ranges.append(length_range)
            metrics_data.append([
                metrics.get('bleu_4', 0.0),
                metrics.get('meteor', 0.0),
                metrics.get('rouge_l', 0.0)
            ])
        
        # 转换为numpy数组
        data = np.array(metrics_data).T
        
        # 创建热力图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
        
        # 设置刻度
        ax.set_xticks(np.arange(len(length_ranges)))
        ax.set_yticks(np.arange(3))
        ax.set_xticklabels(length_ranges)
        ax.set_yticklabels(['BLEU-4', 'METEOR', 'ROUGE-L'])
        
        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # 在每个单元格中显示数值
        for i in range(3):
            for j in range(len(length_ranges)):
                text = ax.text(j, i, f'{data[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=10)
        
        ax.set_title('Performance Metrics Heatmap by Code Length', 
                    fontsize=12, fontweight='bold', pad=20)
        ax.set_xlabel('Code Length Range', fontsize=11)
        ax.set_ylabel('Metric', fontsize=11)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Score', rotation=270, labelpad=20, fontsize=11)
        
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Length-performance heatmap saved to: {save_path}")
    
    def generate_sample_outputs(
        self,
        samples: List[Dict[str, Any]],
        num_samples: int = 10,
        save_path: Optional[str] = None
    ) -> None:
        """
        随机选择测试样例展示原始代码、参考摘要和生成摘要
        
        Args:
            samples: 样例列表，每个样例包含code, reference, generated
            num_samples: 要展示的样例数量
            save_path: 保存路径
        """
        self.logger.info(f"Generating sample outputs (showing {num_samples} samples)...")
        
        # 如果没有指定保存路径，使用默认路径
        if save_path is None:
            output_dir = self.config.get("experiment.output_dir", "./outputs")
            save_path = os.path.join(output_dir, "sample_outputs.txt")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 随机选择样例
        if len(samples) > num_samples:
            selected_samples = random.sample(samples, num_samples)
        else:
            selected_samples = samples
            self.logger.warning(f"Only {len(samples)} samples available, showing all")
        
        # 生成格式化的输出
        output_lines = []
        output_lines.append("=" * 80)
        output_lines.append("SAMPLE OUTPUTS")
        output_lines.append("=" * 80)
        output_lines.append("")
        
        for i, sample in enumerate(selected_samples, 1):
            output_lines.append(f"Sample {i}/{len(selected_samples)}")
            output_lines.append("-" * 80)
            
            # 原始代码
            output_lines.append("Original Code:")
            code = sample.get('code', '')
            # 限制代码长度以便阅读
            if len(code) > 500:
                code = code[:500] + "\n... (truncated)"
            output_lines.append(code)
            output_lines.append("")
            
            # 参考摘要
            output_lines.append("Reference Summary:")
            output_lines.append(sample.get('reference', ''))
            output_lines.append("")
            
            # 生成的摘要
            output_lines.append("Generated Summary:")
            output_lines.append(sample.get('generated', ''))
            output_lines.append("")
            
            # 代码长度信息
            code_length = sample.get('code_length', len(sample.get('code', '')))
            output_lines.append(f"Code Length: {code_length} characters")
            output_lines.append("")
            output_lines.append("=" * 80)
            output_lines.append("")
        
        # 保存到文件
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        
        self.logger.info(f"Sample outputs saved to: {save_path}")
    
    def create_sample_comparison_table(
        self,
        samples: List[Dict[str, Any]],
        num_samples: int = 10,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        创建样例对比表格（更紧凑的格式）
        
        Args:
            samples: 样例列表
            num_samples: 要展示的样例数量
            save_path: 保存路径
            
        Returns:
            pd.DataFrame: 样例对比表格
        """
        self.logger.info(f"Creating sample comparison table...")
        
        # 如果没有指定保存路径，使用默认路径
        if save_path is None:
            output_dir = self.config.get("experiment.output_dir", "./outputs")
            save_path = os.path.join(output_dir, "sample_comparison.csv")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 随机选择样例
        if len(samples) > num_samples:
            selected_samples = random.sample(samples, num_samples)
        else:
            selected_samples = samples
        
        # 创建表格数据
        table_data = []
        for i, sample in enumerate(selected_samples, 1):
            # 截断代码以便在表格中显示
            code = sample.get('code', '')
            if len(code) > 100:
                code = code[:100] + "..."
            
            table_data.append({
                'Sample ID': i,
                'Code (truncated)': code,
                'Reference Summary': sample.get('reference', ''),
                'Generated Summary': sample.get('generated', ''),
                'Code Length': sample.get('code_length', len(sample.get('code', '')))
            })
        
        # 创建DataFrame
        df = pd.DataFrame(table_data)
        
        # 保存为CSV
        df.to_csv(save_path, index=False, encoding='utf-8')
        self.logger.info(f"Sample comparison table saved to: {save_path}")
        
        return df
    
    def save_figure(
        self,
        fig: plt.Figure,
        filename: str,
        dpi: int = 300,
        formats: Optional[List[str]] = None
    ) -> None:
        """
        将图表保存为高分辨率PNG格式（支持多种格式）
        
        Args:
            fig: matplotlib图表对象
            filename: 文件名（不含扩展名）
            dpi: 分辨率（默认300）
            formats: 保存格式列表，默认为['png']
        """
        if formats is None:
            formats = ['png']
        
        output_dir = self.config.get("experiment.output_dir", "./outputs")
        
        for fmt in formats:
            save_path = os.path.join(output_dir, f"{filename}.{fmt}")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight', format=fmt)
            self.logger.info(f"Figure saved to: {save_path}")
        
        plt.close(fig)
    
    def generate_full_report(
        self,
        all_results: Dict[str, Any],
        training_history: Optional[Dict[str, List[float]]] = None
    ) -> None:
        """
        生成完整的实验报告（包括所有表格和图表）
        
        Args:
            all_results: 完整的评估结果，包含：
                - overall_metrics: 整体指标
                - metrics_by_length: 按长度分组的指标
                - sample_outputs: 样例输出
                - model_name: 模型名称
                - dataset_size: 数据集大小
            training_history: 训练历史（可选）
        """
        self.logger.info("=" * 60)
        self.logger.info("Generating full experiment report...")
        self.logger.info("=" * 60)
        
        output_dir = self.config.get("experiment.output_dir", "./outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 生成性能对比表格
        self.logger.info("1. Creating performance comparison table...")
        try:
            self.create_performance_table(
                all_results,
                output_formats=['csv', 'latex', 'markdown']
            )
        except Exception as e:
            self.logger.error(f"Failed to create performance table: {e}")
        
        # 2. 生成训练过程可视化（如果有训练历史）
        if training_history:
            self.logger.info("2. Plotting training curves...")
            try:
                self.plot_training_curves(training_history)
                self.create_learning_rate_plot(training_history)
            except Exception as e:
                self.logger.error(f"Failed to plot training curves: {e}")
        else:
            self.logger.warning("No training history provided, skipping training curves")
        
        # 3. 生成代码长度影响分析
        if 'metrics_by_length' in all_results:
            self.logger.info("3. Plotting code length analysis...")
            try:
                self.plot_length_analysis(all_results['metrics_by_length'])
                self.create_length_performance_heatmap(all_results['metrics_by_length'])
            except Exception as e:
                self.logger.error(f"Failed to plot length analysis: {e}")
        else:
            self.logger.warning("No length metrics provided, skipping length analysis")
        
        # 4. 生成样例展示
        if 'sample_outputs' in all_results:
            self.logger.info("4. Generating sample outputs...")
            try:
                num_samples = self.config.get("evaluation.num_samples", 10)
                self.generate_sample_outputs(
                    all_results['sample_outputs'],
                    num_samples=num_samples
                )
                self.create_sample_comparison_table(
                    all_results['sample_outputs'],
                    num_samples=num_samples
                )
            except Exception as e:
                self.logger.error(f"Failed to generate sample outputs: {e}")
        else:
            self.logger.warning("No sample outputs provided, skipping sample generation")
        
        # 5. 生成实验摘要报告
        self.logger.info("5. Creating experiment summary...")
        try:
            self._create_experiment_summary(all_results, training_history)
        except Exception as e:
            self.logger.error(f"Failed to create experiment summary: {e}")
        
        self.logger.info("=" * 60)
        self.logger.info("Full report generation completed!")
        self.logger.info(f"All outputs saved to: {output_dir}")
        self.logger.info("=" * 60)
    
    def _create_experiment_summary(
        self,
        results: Dict[str, Any],
        training_history: Optional[Dict[str, List[float]]] = None
    ) -> None:
        """
        创建实验摘要报告
        
        Args:
            results: 评估结果
            training_history: 训练历史
        """
        output_dir = self.config.get("experiment.output_dir", "./outputs")
        summary_path = os.path.join(output_dir, "experiment_summary.txt")
        
        lines = []
        lines.append("=" * 80)
        lines.append("EXPERIMENT SUMMARY")
        lines.append("=" * 80)
        lines.append("")
        
        # 基本信息
        lines.append("Basic Information:")
        lines.append(f"  Model: {results.get('model_name', 'Unknown')}")
        lines.append(f"  Dataset Size: {results.get('dataset_size', 0)}")
        lines.append(f"  Experiment Name: {self.config.get('experiment.name', 'Unknown')}")
        lines.append("")
        
        # 整体性能
        if 'overall_metrics' in results:
            lines.append("Overall Performance:")
            metrics = results['overall_metrics']
            lines.append(f"  BLEU-4:  {metrics.get('bleu_4', 0.0):.4f}")
            lines.append(f"  METEOR:  {metrics.get('meteor', 0.0):.4f}")
            lines.append(f"  ROUGE-L: {metrics.get('rouge_l', 0.0):.4f}")
            lines.append("")
        
        # 按长度分组的性能
        if 'metrics_by_length' in results:
            lines.append("Performance by Code Length:")
            for length_range, metrics in sorted(results['metrics_by_length'].items()):
                lines.append(f"  {length_range}:")
                lines.append(f"    Sample Count: {metrics.get('sample_count', 0)}")
                lines.append(f"    BLEU-4:  {metrics.get('bleu_4', 0.0):.4f}")
                lines.append(f"    METEOR:  {metrics.get('meteor', 0.0):.4f}")
                lines.append(f"    ROUGE-L: {metrics.get('rouge_l', 0.0):.4f}")
            lines.append("")
        
        # 训练信息
        if training_history:
            lines.append("Training Information:")
            if 'train_loss' in training_history and training_history['train_loss']:
                lines.append(f"  Final Training Loss: {training_history['train_loss'][-1]:.4f}")
            if 'eval_loss' in training_history and training_history['eval_loss']:
                lines.append(f"  Best Validation Loss: {min(training_history['eval_loss']):.4f}")
            if 'learning_rate' in training_history and training_history['learning_rate']:
                lines.append(f"  Final Learning Rate: {training_history['learning_rate'][-1]:.2e}")
            lines.append("")
        
        # 配置信息
        lines.append("Configuration:")
        lines.append(f"  Learning Rate: {self.config.get('model.learning_rate', 'N/A')}")
        lines.append(f"  Batch Size: {self.config.get('model.batch_size', 'N/A')}")
        lines.append(f"  Epochs: {self.config.get('model.num_epochs', 'N/A')}")
        lines.append(f"  Beam Size: {self.config.get('evaluation.beam_size', 'N/A')}")
        lines.append("")
        
        lines.append("=" * 80)
        
        # 保存摘要
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        self.logger.info(f"Experiment summary saved to: {summary_path}")

