# PhisHGMAE: 基于异构图掩码自编码器的钓鱼网站检测

## 概述
PhisHGMAE是一个基于深度学习的钓鱼网站检测项目，利用异构图掩码自编码器技术。该模型采用基于元路径的方法来学习URL关系和特征的有效表示，从而准确地对钓鱼和合法网站进行分类。

## 特点
- 基于异构图的钓鱼检测方法
- 掩码自编码器架构，用于稳健的特征学习
- 实现MetaPath2Vec用于基于元路径的节点嵌入
- 支持多种元路径和异构图结构
- 分类任务，包含全面的评估指标

## 环境要求
- Python 3.12+
- PyTorch 2.3.1
- PyTorch Geometric 2.6.1
- Scikit-learn
- Loguru

## 安装

### 使用uv（推荐）
PhisHGMAE使用[uv](https://github.com/astral-sh/uv)作为其包管理和虚拟环境系统。

1. 安装uv（如果尚未安装）：
```bash
pip install uv
```

2. 克隆仓库：
```bash
git clone <仓库链接>
cd PhisHGMAE
```

3. 创建虚拟环境并安装依赖：
```bash
uv venv
uv pip install -e .
```

## 使用方法

### 数据准备
按照适当的格式准备数据集，并将其放在`data/`目录中。系统期望异构图数据具有特定的元路径结构。

### 配置
编辑`configs.yml`文件设置模型超参数和实验配置。配置包括以下参数：

- 模型架构（隐藏维度、层数等）
- 训练设置（学习率、迭代次数等）
- MetaPath2Vec配置
- 自编码器的掩码率

### 训练和评估
要训练模型并评估其性能：

```bash
python main.py --task classification --use_cfg
```

对于基于MLP的不同训练比例的评估：
```bash
python evaluate_phishgmae.py
```

### 输出
- 训练好的模型嵌入将保存到`embeddings/`目录
- 评估结果将显示在控制台上并保存到`results/`目录
- 日志存储在`logs/`目录

## 项目结构
- `main.py`：训练和评估的主入口点
- `models/`：包含模型架构定义
  - `edcoder.py`：编码器-解码器架构
  - `han.py`：分层注意力网络实现
  - `gat.py`：图注意力网络实现
- `utils/`：实用函数和辅助工具
- `data/`：存储数据集的目录
- `configs.yml`：实验配置文件
- `evaluate_phishgmae.py`：用于基于MLP的评估脚本
