# LLM 笑话理解与分类评估框架

本项目是一个用于全面评估大语言模型（LLM）在理解、分类和解释冒犯性笑话方面能力的自动化框架。

该框架通过多维度评分体系，从**分类准确性**和**深度理解**两个层面，对模型的能力进行量化分析，并提供丰富的可视化结果。

## 项目目标

- **标准化评估**: 提供一个可复现、可扩展的流程，用于评估不同 LLM 在处理具有复杂文化和逻辑背景的文本（如冒犯性笑话）时的表现。
- **多维度分析**: 不仅评估模型对攻击性等级、类别和目标的**分类能力**，还通过引入 LLM 作为裁判，深度评估其对笑话背后**背景知识**和**幽默逻辑**的**理解能力**。
- **自动化与工程化**: 所有评估和分析流程都通过参数化的命令行脚本实现，支持轻松配置和批量处理。

## 框架结构

```
.
├── data/
│   ├── label_jokes.jsonl         # 包含笑话、ID、真实背景和逻辑解释的基准数据
│   └── offensive - Sheet1.csv    # 原始笑话数据
├── result/
│   ├── evaluation/               # 各模型分类结果的输出目录
│   │   └── [model_name]/
│   │       ├── detailed_results.csv
│   │       └── analyzed_jokes.jsonl
│   ├── scoring/                  # 各模型评分分析结果的输出目录
│   │   └── [model_name]/
│   │       ├── detailed_scores.csv
│   │       ├── scoring_summary.json
│   │       └── [model_name]_scores_visualization.png
│   └── comparison/               # 多模型横向对比结果
│       ├── all_models_scores.csv
│       ├── model_comparison.png
│       └── score_correlation_heatmap.png
├── src/
│   ├── data_process/
│   │   └── analyze_jokes.py      # 步骤1: 使用目标模型分析笑话，生成背景和逻辑解释
│   └── evaluation/
│       ├── evaluate_model.py     # 步骤2: 评估模型在分类任务上的表现
│       ├── score_analyzer.py     # 步骤3: 对单个模型进行全面的五维度评分和分析
│       └── compare_scores.py     # 步骤4: 对所有已评分的模型进行横向对比
└── openai_key.txt                # OpenAI API 密钥和 Base URL 配置文件
```

## 核心功能与脚本

### 1. `src/data_process/analyze_jokes.py`

- **功能**: 使用指定的大模型（如 `deepseek-v3`）读取原始笑话，并为每个笑话生成其背后的**背景（background）**和**逻辑（logic）**解释。
- **输出**: 为每个模型在 `result/evaluation/[model_name]/` 目录下生成一个 `analyzed_jokes.jsonl` 文件。

### 2. `src/evaluation/evaluate_model.py`

- **功能**: 评估模型在笑话**分类**任务上的表现，将其预测的 `label`, `category`, `target` 与真实标签进行对比。
- **输出**: 为每个模型在 `result/evaluation/[model_name]/` 目录下生成一个 `detailed_results.csv` 文件。

### 3. `src/evaluation/score_analyzer.py`

- **功能**: 框架的核心。对单个模型进行全面的**五维度评分**。
    - **数据整合**: 自动加载并合并 `detailed_results.csv`、`analyzed_jokes.jsonl` 和 `data/label_jokes.jsonl`。
    - **维度1-3 (分类评分)**:
        - `label_score`: 冒犯等级评分（准确得3分，偏差1级得1分）。
        - `category_score`: 冒犯类别评分（准确得1分）。
        - `target_score`: 攻击目标评分（准确得1分）。
    - **维度4-5 (理解力评分)**:
        - 使用一个强大的裁判模型（如 GPT-4o）并行调用 API，根据预设的中文评分标准，对模型生成的 `background` 和 `logic` 进行评分（0-3分）。
- **输出**:
    - `detailed_scores.csv`: 包含所有原始数据、预测和五维度分数的详细表格。
    - `scoring_summary.json`: 包含平均分和每个维度最差表现案例的总结。
    - `[model_name]_scores_visualization.png`: 模型在各维度平均分的条形图。

### 4. `src/evaluation/compare_scores.py`

- **功能**: 对所有已完成评分的模型进行横向对比。
- **输出**:
    - `all_models_scores.csv`: 汇总所有模型平均分的数据表。
    - `model_comparison.png`: 分组条形图，直观比较各模型表现。
    - `score_correlation_heatmap.png`: 相关性热力图，分析各评分维度之间的关系。

## 使用方法

### 步骤 0: 准备工作

1.  **配置API密钥**: 在项目根目录创建 `openai_key.txt` 文件，并按以下格式填入你的 API Key 和 Base URL：
    ```
    OPENAI_API_KEY: sk-xxxxxxxxxx
    OPENAI_BASE_URL: https://api.example.com/v1
    ```
2.  **准备数据**: 确保 `data/label_jokes.jsonl` 和 `data/offensive - Sheet1.csv` 文件存在且格式正确。

### 步骤 1: 分析笑话 (生成背景和逻辑)

为 `deepseek-v3` 模型生成解释：
```bash
python src/data_process/analyze_jokes.py --model_name deepseek-v3
```

### 步骤 2: 评估分类表现

评估 `deepseek-v3` 模型的分类准确性：
```bash
python src/evaluation/evaluate_model.py --model_name deepseek-v3
```

### 步骤 3: 进行全面评分和分析

对 `deepseek-v3` 模型进行五维度评分：
```bash
python src/evaluation/score_analyzer.py --model_name deepseek-v3
```

### 步骤 4: 横向比较所有模型

在对一个或多个模型完成评分后，运行比较脚本：
```bash
python src/evaluation/compare_scores.py
```
此脚本会自动查找所有位于 `result/scoring/` 下的评分总结，并生成对比图表。

## 如何扩展

要评估一个新的模型（例如 `qwen-v2`），只需重复步骤 1-3，并将 `--model_name` 参数替换为新模型的名称即可：

1.  `python src/data_process/analyze_jokes.py --model_name qwen-v2`
2.  `python src/evaluation/evaluate_model.py --model_name qwen-v2`
3.  `python src/evaluation/score_analyzer.py --model_name qwen-v2`

完成以上步骤后，再次运行 `compare_scores.py`，新的模型就会自动出现在比较结果中。
