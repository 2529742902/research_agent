# research_agent
# Research Agent 自动化互联网调研双智能体系统

本项目是一个基于 **LangGraph React Agent** 架构搭建的自动化互联网调研与评测闭环系统。系统通过双智能体（Research Agent & Critic Agent）协同工作：首先由 Research Agent 全网检索并深度抓取核心信息，自动交叉验证后输出严谨的 Markdown 调研报告；随后由 Critic Agent 担任评审专家对报告进行多维度打分和合规性审查，并基于诊断结果**自动定向优化系统 Prompt**，实现智能体策略的自我迭代。

## 项目特点与核心逻辑

* **双模型分工协作**：
* **Research LLM (`deepseek-v4-flash`)**：专注于高频工具调用、全网多源文本抓取与结构化报告撰写，追求极致性价比与响应速度。


* **Critic LLM (`deepseek-v4-pro` / DeepSeek-R1 推理模型)**：发挥深度逻辑推理优势，专注于多维度打分、文本规则匹配与系统 Prompt 定向优化，裁判模型温度设为 `0.0` 确保评测标准一致。




* **多源交叉验证**：系统强制要求关键数据必须来自两个及以上的独立来源。如果遭遇分歧，智能体会自动在报告中如实标注并提供各自出处，彻底杜绝大模型的模糊虚假表述。


* **自动化 Prompt 优化闭环**：Critic Agent 根据评估出的薄弱环节（如信息覆盖度低、来源可追溯性差、逻辑连贯性不足等），基于内置专家规则定向调整 Prompt，自动生成改进后的新系统指令并保存至本地 `optimized_prompt.txt`。



## 项目文件结构

* `main.py`：系统执行总入口，构建并串联主流程管道。


* `agent.py`：负责加载环境变量并初始化各阶段大语言模型客户端。


* `research_agent_tool.py`：提供全网检索（Tavily API）、抗噪网页深度抓取（BeautifulSoup4）、arXiv 学术接口对接及本地文件持久化读写工具箱。


* `eval_tools.py`：提供评审打分（LLM-as-Judge）、引用来源可追溯性规则匹配、运行效率耗时统计以及 Prompt 定向优化器工具箱。


* `requirements.txt`：项目依赖包列表。



## 快速开始

### 1. 安装依赖

在项目根目录下打开终端，一键安装所有必需依赖：

```bash
pip install -r requirements.txt

```

### 2. 配置环境变量

在项目根目录下新建 `.env` 配置文件，并配置你的 API 密钥：

```text
DeepSeek_API_KEY=your_deepseek_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

```

### 3. 启动运行

执行主程序，并根据终端提示输入你想调研的主题（例如：`Transformer 推理加速技术最新进展`）：

```bash
python main.py

```

### 4. 查看输出成果

项目运行完成后，系统会在根目录下自动生成两份核心成果：

* **`final_report.md`**：Research Agent 最终撰写输出的严谨 Markdown 调研报告，包含优劣势对比表格、趋势判断、具体落地建议以及可追溯的参考文献链接。


* **`optimized_prompt.txt`**：Critic Agent 结合本次报告表现及运行效率指标，自动优化并迭代出的最新版系统 Prompt 提示词。
