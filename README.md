# 基于深度学习的代码摘要生成实验

本项目实现了基于CodeT5模型的代码摘要自动生成系统，使用CodeSearchNet Python数据集进行训练和评估。

## 项目概述

代码摘要生成（Code Summarization）是软件工程中的重要任务，旨在自动生成代码片段的自然语言描述。本项目采用Transformer架构的CodeT5-small模型，通过序列到序列（Seq2Seq）的端到端学习范式，实现从源代码到自然语言描述的自动转换。

## 实验结果

- **训练样本**: 80,000
- **训练时间**: 68分18秒
- **训练速度**: 3.0 iterations/秒
- **最终训练损失**: 0.0033
- **最终验证损失**: 0.0074
- **收敛效果**: 损失从8.61快速降至0.003，展现良好的学习能力

详细的实验结果和分析请参见 [实验报告.md](实验报告.md)

## 环境要求

### 硬件环境
- GPU: NVIDIA GeForce RTX 4060 (8GB) 或同等性能
- 内存: 16GB+
- 存储: 至少10GB可用空间

### 软件环境
```
Python 3.12.6
PyTorch 2.5.1
Transformers 4.57.3
CUDA 12.1
```

完整依赖列表见 [requirements.txt](requirements.txt)

## 安装步骤

1. 克隆项目
```bash
git clone <repository-url>
cd 期末作业
```

2. 创建虚拟环境（推荐）
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

## 数据准备

### 方式1: 自动下载（推荐）
运行主程序时会自动下载CodeSearchNet数据集：
```bash
python main.py --config config/experiment.yaml
```

### 方式2: 使用本地数据
如果已有处理好的数据，修改配置文件中的路径：
```yaml
data:
  local_processed_data: "./data/processed"
```

## 使用方法

### 快速开始

运行完整流程（数据处理 + 训练 + 评估）：
```bash
python main.py --config config/experiment.yaml
```

### 仅训练
```bash
python main.py --config config/experiment.yaml --mode train
```

### 仅评估
```bash
python main.py --config config/experiment.yaml --mode evaluate
```

### 仅数据处理
```bash
python main.py --config config/experiment.yaml --mode process_data
```

## 配置说明

主要配置文件: `config/experiment.yaml`

### 关键参数

**数据配置**:
```yaml
data:
  train_size: 80000        # 训练样本数
  val_size: 5000           # 验证样本数
  test_size: 5000          # 测试样本数
  max_source_length: 256   # 代码最大长度
  max_target_length: 128   # 摘要最大长度
```

**模型配置**:
```yaml
model:
  name: "Salesforce/codet5-small"
  learning_rate: 0.00005
  batch_size: 32
  num_epochs: 5
  fp16: true                      # 混合精度训练
  dataloader_num_workers: 0       # Windows必须为0
```

**优化建议**:
- Windows环境: `dataloader_num_workers: 0`
- 启用混合精度: `fp16: true` (节省显存，加速训练)
- 根据GPU显存调整 `batch_size`

## 项目结构

```
期末作业/
├── config/                      # 配置文件
│   ├── default.yaml            # 默认配置
│   └── experiment.yaml         # 实验配置
├── data/                        # 数据目录
│   ├── raw/                    # 原始数据
│   └── processed/              # 处理后的数据
├── src/                         # 源代码
│   ├── config/                 # 配置管理
│   ├── data/                   # 数据处理
│   ├── models/                 # 模型定义
│   ├── evaluation/             # 评估模块
│   ├── utils/                  # 工具函数
│   └── visualization/          # 可视化
├── outputs/                     # 实验输出
│   ├── best_model/             # 最佳模型
│   ├── checkpoint-*/           # 训练检查点
│   ├── experiment_summary.json # 实验摘要
│   ├── training_history.json   # 训练历史
│   └── config_backup.yaml      # 配置备份
├── logs/                        # 日志文件
├── tests/                       # 测试代码
├── main.py                      # 主程序入口
├── requirements.txt             # 依赖列表
├── 实验报告.md                  # 详细实验报告
├── 期末作业.md                  # 作业要求
├── 期末作业.pdf                 # 作业要求PDF
└── README.md                    # 本文件
```

## 输出说明

训练完成后，输出目录 `outputs/` 包含：

- `best_model/`: 最佳模型权重
- `checkpoint-*/`: 训练检查点
- `experiment_summary.json`: 实验摘要（时间、指标、资源使用）
- `training_history.json`: 完整训练历史（每步的loss和学习率）
- `config_backup.yaml`: 配置备份

## 性能优化

### 训练速度优化
1. **混合精度训练**: 启用 `fp16: true`，速度提升约30%
2. **数据加载**: Windows环境设置 `dataloader_num_workers: 0`，避免多进程开销
3. **批次大小**: 根据GPU显存调整，RTX 4060建议32
4. **梯度累积**: 显存不足时使用 `gradient_accumulation_steps`

### 显存优化
- 降低 `batch_size`
- 减少 `max_source_length` 和 `max_target_length`
- 启用 `fp16` 混合精度
- 使用梯度检查点（需修改代码）

## 常见问题

### Q: 训练速度很慢（<1 it/s）
A: 检查以下配置：
- Windows环境必须设置 `dataloader_num_workers: 0`
- 启用混合精度 `fp16: true`
- 确认GPU正常工作

### Q: 显存不足（CUDA out of memory）
A: 
- 降低 `batch_size`（如32→16→8）
- 减少序列长度
- 启用 `fp16`

### Q: 数据下载失败
A: 
- 检查网络连接
- 使用代理或VPN
- 手动下载数据集后放到 `data/raw/` 目录

### Q: 模型效果不理想
A: 
- 增加训练样本数
- 增加训练轮数
- 调整学习率
- 尝试更大的模型（codet5-base）

## 评估指标

本项目使用以下指标评估模型性能：

- **BLEU-4**: 衡量生成摘要与参考摘要的n-gram重叠度
- **METEOR**: 考虑同义词和词干的匹配度
- **ROUGE-L**: 基于最长公共子序列的相似度

参考性能（10K样本小规模实验）：
- BLEU-4: 55.51%
- METEOR: 64.26%
- ROUGE-L: 74.08%

## 扩展实验

### 使用完整数据集
修改配置文件中的样本数：
```yaml
data:
  train_size: 251820  # 完整训练集
  val_size: 13914     # 完整验证集
  test_size: 14918    # 完整测试集
```

预计训练时间：约4-5小时（RTX 4060）

### 使用更大模型
```yaml
model:
  name: "Salesforce/codet5-base"  # 220M参数
```

注意：需要更多显存和训练时间

## 测试

运行单元测试：
```bash
pytest tests/
```

运行特定测试：
```bash
pytest tests/test_data_processor.py
pytest tests/test_trainer.py
```

## 引用

如果使用本项目，请引用：

```bibtex
@misc{code-summarization-2026,
  title={基于深度学习的代码摘要生成实验},
  author={Your Name},
  year={2026},
  howpublished={\url{<repository-url>}}
}
```

CodeT5模型：
```bibtex
@inproceedings{wang2021codet5,
  title={CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation},
  author={Wang, Yue and Wang, Weishi and Joty, Shafiq and Hoi, Steven CH},
  booktitle={EMNLP},
  year={2021}
}
```

## 许可证

本项目仅用于学术研究和教育目的。

## 联系方式

如有问题或建议，请提交Issue或联系项目维护者。

## 致谢

- Salesforce团队的CodeT5模型
- Hugging Face的Transformers库
- CodeSearchNet数据集项目
