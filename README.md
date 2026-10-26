# RAG-Agent-Eval-Kit

![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

**RAG-Agent-Eval-Kit** 是一个面向大模型应用（RAG & Agent）的模块化、可扩展自动化评测框架。旨在解决 RAG 系统中“检索与生成故障难定位”以及 Agent 系统“工具调用轨迹难量化”的核心痛点。

## 🎯 核心特性

- **分层解耦架构**：采用 Adapter 设计模式，将评测引擎（RAGAS/DeepEval）与业务调度彻底分离。
- **RAG 深度诊断**：已集成业界主流框架 **RAGAS**，支持 `Faithfulness`、`Answer Relevancy`、`Context Precision`、`Context Recall` 四大核心指标。
- **物理级依赖隔离**：基于 **Docker 多阶段构建（Multi-stage Build）**，彻底解决不同评测框架（如 RAGAS 与 DeepEval）之间的依赖冲突地狱。
- **全本地化部署**：完美支持通过本地 **Ollama**（如 Qwen2.5）和本地向量模型（如 `nomic-embed-text`）作为裁判，无需强依赖 OpenAI API。

## 🗺️ 演进路线 (Roadmap)

- [x] 核心架构搭建与依赖隔离机制
- [x] 集成 RAGAS 评测引擎（基线跑通）
- [ ] 集成 DeepEval，实现 CI/CD 自动化门禁
- [ ] 开发 Agent 轨迹评测（工具调用准确率、步数效率）
- [ ] 集成可视化图表生成（雷达图、柱状图）

## 📂 项目结构

```text
RAG-Agent-Eval-Kit/
├── assets/                 # 演示截图与资源
├── datasets/               # 统一测试数据集 (JSON 格式)
├── outputs/                # 评测报告输出目录
├── src/
│   ├── adapters/           # 评测组件层 (纯代码，无状态)
│   │   └── ragas_adapter.py
│   ├── scripts/            # 运行入口层 (读取配置，调度 Adapter)
│   │   └── run_ragas_eval.py
│   ├── core/               # 核心业务逻辑
│   └── utils/              # 工具函数
├── requirements-ragas.txt  # 隔离的依赖文件
└── docker-compose.yml      # 多环境编排

📊 RAGAS 基线评测结果
以下为本地 Qwen2.5:7b 模型结合 nomic-embed-text 向量模型，在 10 条中文 RAG 测试集上的真实跑分：

https://assets/images/ragas_result.png

💡 评测洞察 (Baseline Analysis)
指标	分数	解读
Faithfulness (忠实度)	1.0	满分表现，模型回答完全基于给定上下文，无幻觉。
Context Precision (精确率)	1.0	满分表现，检索到的上下文高度相关，无冗余信息。
Context Recall (召回率)	1.0	满分表现，检索环节成功覆盖了所有标准答案所需的信息。
Answer Relevancy (相关性)	0.79	表现良好，但存在部分回答略显啰嗦或未直接切中要害的问题，是后续优化的重点。
工程突破：本项目通过 Monkey Patch 技术，完美修复了 RAGAS 0.4.3 版本在导入 ChatVertexAI 时引发的底层依赖 Bug，确保了评测流水线的高可用性。

🚀 快速开始
1. 准备本地模型服务
请确保本地已运行 Ollama，并下载以下模型：

bash
ollama run qwen2.5:7b
ollama pull nomic-embed-text
2. 配置环境
bash
# 建议使用 Python 3.11 环境
python -m venv venv_ragas
source venv_ragas/bin/activate  # Windows: .\venv_ragas\Scripts\activate

# 安装专用依赖
pip install -r requirements-ragas.txt
3. 运行评测
bash
python src/scripts/run_ragas_eval.py
运行结果将自动保存至 outputs/ragas_results.json。

⚠️ 免责声明
本仓库中的测试数据集仅供评测框架的验证与学术研究使用。

📄 开源协议
本项目采用 Apache License 2.0 协议开源。