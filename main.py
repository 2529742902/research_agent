import os
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from agent import research_llm, critic_llm
from research_agent_tool import research_tools
from eval_tools import eval_tools, start_tracking

start_tracking()

# ==========================================
# 系统 Prompt
# ==========================================
research_system_prompt = """你是一个专业的互联网调研 Agent。
你的任务是根据用户的调研主题，综合运用网络搜索、网页抓取、学术文献检索等工具，最终撰写一份结构化的 Markdown 调研报告并保存到本地。

调研流程建议：
1. 先用 advanced_web_search 搜索主题概览，识别主要观点、关键数据、权威来源和争议点。
2. 对重要链接用 smart_web_scraper 深度抓取。对于抓取到的任何关键数据、统计结果、引用结论，必须：
   - 记录该信息所在的原始页面 URL；
   - 通过至少一个其他独立来源进行交叉验证，确认一致后再采纳；
   - 如果两个以上独立来源存在矛盾，必须在报告中如实说明分歧，并注明各自出处。
3. 若涉及技术/算法，用 academic_literature_search 检索相关论文，引用论文时需列出 DOI 或可直接访问的官方URL。
4. 用 local_file_manager 将最终报告写入 final_report.md。
---

报告撰写强制要求（必须严格遵守）：
- 所有关键事实、数据、引述必须附带原始来源 URL，放置于对应段落的括号或脚注中。不允许出现“研究表明”“数据显示”等无出处的模糊表述。
- 整个报告的事实基础必须达到交叉验证标准：关键数据至少来自两个独立来源，否则不能作为确定性结论写入报告；若只能找到单一来源，必须在文中明确标注“单一来源，待进一步验证”。
- 报告结论部分必须包含以下三项内容，缺一不可：
    (a) 各主要方案/观点的优劣势对比表格，列明方案名称、核心优势、主要劣势、适用场景、数据支撑来源；
    (b) 基于所得数据的趋势判断，明确指出未来发展方向或演变路径，并说明判断依据；
    (c) 至少一条针对特定场景的具体、可落地的建议，该建议必须与前述数据与对比有直接逻辑关联，不得泛泛而谈。
- 结论部分严禁仅对前文信息进行复述或摘要，必须体现独立的分析、对比与推断。
- 报告整体应结构清晰，使用 Markdown 标题、表格、列表等元素提升可读性，但不得因格式而牺牲内容的准确性和深度。

请始终以严谨、客观、可验证的方式执行调研与写作。
"""

critic_system_prompt = """你是一个严格、公正的调研报告评审专家。
请按照以下顺序依次完成评测任务：

1. 用 llm_as_judge 对报告进行四维度打分，保存返回的 JSON 结果
2. 用 source_traceability_checker 检查报告中的引用来源合规性
3. 用 get_run_statistics 获取本次运行的效率统计
4. 用 prompt_optimizer，传入上面 llm_as_judge 返回的 JSON 和当前 Research Prompt，生成优化建议

请逐步完成，不要跳过任何步骤。
"""

# ==========================================
# 创建两个 Agent
# ==========================================
research_agent = create_react_agent(
    model=research_llm,
    tools=research_tools,
    prompt=research_system_prompt
)

critic_agent = create_react_agent(
    model=critic_llm,
    tools=eval_tools,
    prompt=critic_system_prompt
)


# ==========================================
# 主流程：串联两个 Agent
# ==========================================
def run_pipeline(topic: str):

    # ---------- 阶段一：Research Agent ----------
    print("\n" + "=" * 50)
    print("📚 阶段一：Research Agent 开始调研")
    print("=" * 50 + "\n")

    start_tracking()  # 开始计时

    for chunk in research_agent.stream(
        {"messages": [HumanMessage(content=topic)]},
        stream_mode="values"
    ):
        last_msg = chunk["messages"][-1]
        last_msg.pretty_print()

    print("\n" + "=" * 50)
    print("✅ 阶段一结束：调研报告已生成")
    print("=" * 50)

    # ---------- 读取报告内容，传给 Critic ----------
    report_path = "final_report.md"
    if not os.path.exists(report_path):
        print(f"\n❌ 未找到报告文件 '{report_path}'，Critic Agent 无法启动。")
        print("请检查 Research Agent 是否成功调用了 local_file_manager 写入报告。")
        return

    with open(report_path, "r", encoding="utf-8") as f:
        report_content = f.read()

    # ---------- 阶段二：Critic Agent ----------
    print("\n" + "=" * 50)
    print("🔍 阶段二：Critic Agent 开始评测")
    print("=" * 50 + "\n")

    # 把报告内容、原始问题、当前 Prompt 一起传给 Critic Agent
    critic_input = f"""
请对以下调研报告进行完整评测。

【原始调研问题】
{topic}

【当前 Research Agent 系统 Prompt】
{research_system_prompt}

【待评测报告内容】
{report_content}
"""

    for chunk in critic_agent.stream(
        {"messages": [HumanMessage(content=critic_input)]},
        stream_mode="values"
    ):
        last_msg = chunk["messages"][-1]
        last_msg.pretty_print()

    print("\n" + "=" * 50)
    print("✅ 阶段二结束：评测完成")
    print("=" * 50)
    print("\n💡 如需使用优化后的 Prompt，请查看当前目录下的 optimized_prompt.txt")


# ==========================================
# 入口
# ==========================================
if __name__ == "__main__":
    topic = input("请输入调研主题：")
    run_pipeline(topic)