import os
import re
import time
import json
import requests
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 工具 11：LLM-as-Judge 评测工具
# ==========================================
@tool
def llm_as_judge(report_content: str, research_topic: str) -> str:
    """
    使用 Critic LLM 对调研报告进行多维度自动评分。

    Args:
        report_content: 调研报告的完整 Markdown 文本。
        research_topic: 原始调研问题，用于判断信息覆盖是否到位。

    Returns:
        JSON 格式字符串，包含各维度评分（1-5分）与评分理由，以及总分。
    """
    from agent import critic_llm  # 延迟导入，避免循环依赖

    judge_prompt = f"""你是一位严格、公正的调研报告评审专家。请根据以下四个维度，对报告进行逐项评分。

【原始调研问题】
{research_topic}

【待评审报告】
{report_content}

---

【评分维度说明】

1. 信息覆盖度（1-5分）
   - 5分：完整覆盖调研问题的所有核心方面，无明显遗漏
   - 3分：覆盖主要方面，但有1-2个重要子问题未涉及
   - 1分：仅覆盖表面或单一角度，严重缺失关键信息

2. 事实准确性（1-5分）
   - 5分：所有关键陈述均有来源支撑，无明显错误或过时信息
   - 3分：大部分陈述可信，存在少量未经验证的表述
   - 1分：存在明显错误、自相矛盾或大量无依据的断言

3. 逻辑连贯性（1-5分）
   - 5分：报告结构清晰，各章节衔接自然，论证严密
   - 3分：整体可读，但部分段落跳跃感强或层次混乱
   - 1分：结构混乱，缺乏主线，难以连贯阅读

4. 深度与洞察（1-5分）
   - 5分：超越信息罗列，包含比较分析、趋势判断或独立结论
   - 3分：有一定分析，但结论较浅，缺乏原创性见解
   - 1分：纯粹堆砌事实，无任何分析层次

---

【输出要求】
请严格按照以下 JSON 格式输出，不要添加任何额外说明：

{{
  "scores": {{
    "信息覆盖度": {{"score": <1-5的整数>, "reason": "<简短评分理由，50字以内>"}},
    "事实准确性": {{"score": <1-5的整数>, "reason": "<简短评分理由，50字以内>"}},
    "逻辑连贯性": {{"score": <1-5的整数>, "reason": "<简短评分理由，50字以内>"}},
    "深度与洞察": {{"score": <1-5的整数>, "reason": "<简短评分理由，50字以内>"}}
  }},
  "total_score": <四项之和，满分20>,
  "overall_comment": "<整体评价，100字以内>",
  "weakest_dimension": "<得分最低的维度名称>"
}}"""

    try:
        response = critic_llm.invoke(judge_prompt)
        raw_text = response.content.strip()

        # 提取 JSON 块（防止模型在 JSON 前后加多余文字）
        json_match = re.search(r'\{[\s\S]*\}', raw_text)
        if not json_match:
            return f"评分失败：模型未返回合法 JSON。原始输出：\n{raw_text}"

        result = json.loads(json_match.group())
        return json.dumps(result, ensure_ascii=False, indent=2)

    except json.JSONDecodeError as e:
        return f"评分失败：JSON 解析错误 - {str(e)}\n原始输出：\n{raw_text}"
    except Exception as e:
        return f"评分发生异常：{str(e)}"


# ==========================================
# 工具 12：来源可追溯性检查工具（规则匹配）
# ==========================================
@tool
def source_traceability_checker(report_content: str) -> str:
    """
    对调研报告中的引用来源进行规则匹配检查，统计引用数量、格式合规率，
    并实际访问 URL 验证其可达性。

    Args:
        report_content: 调研报告的完整 Markdown 文本。

    Returns:
        包含引用统计、格式合规率和 URL 可达性检测结果的文本报告。
    """

    # --- Step 1: 提取所有 URL ---
    # 匹配 Markdown 链接格式 [文字](url) 和裸 URL
    md_link_pattern = re.compile(r'\[([^\]]+)\]\((https?://[^\)]+)\)')
    bare_url_pattern = re.compile(r'(?<!\()(https?://[^\s\)\"\']+)')

    md_links = md_link_pattern.findall(report_content)          # [(文字, url), ...]
    md_urls = [url for _, url in md_links]

    all_raw_urls = bare_url_pattern.findall(report_content)
    # 去掉已经在 Markdown 链接里的 URL，避免重复统计
    bare_urls = [u for u in all_raw_urls if u not in md_urls]

    all_urls = list(dict.fromkeys(md_urls + bare_urls))         # 去重，保留顺序
    total_url_count = len(all_urls)
    md_format_count = len(md_urls)

    # --- Step 2: 格式合规率（以 Markdown 超链接格式为"合规"标准）---
    compliance_rate = (md_format_count / total_url_count * 100) if total_url_count > 0 else 0.0

    # --- Step 3: URL 可达性检测 ---
    url_check_results = []
    headers = {"User-Agent": "Mozilla/5.0"}

    for url in all_urls[:10]:   # 最多检测前 10 个，避免耗时过长
        try:
            resp = requests.head(url, headers=headers, timeout=6, allow_redirects=True)
            status = resp.status_code
            reachable = "✅ 可访问" if status < 400 else f"⚠️ 异常（{status}）"
        except requests.exceptions.Timeout:
            reachable = "❌ 超时"
        except requests.exceptions.ConnectionError:
            reachable = "❌ 连接失败"
        except Exception as e:
            reachable = f"❌ 错误（{str(e)[:30]}）"

        url_check_results.append(f"  {reachable} | {url}")

    reachable_count = sum(1 for r in url_check_results if "✅" in r)

    # --- Step 4: 组装输出报告 ---
    lines = [
        "=" * 50,
        "【来源可追溯性检查报告】",
        "=" * 50,
        f"📊 引用 URL 总数：{total_url_count} 个",
        f"📝 Markdown 规范格式引用：{md_format_count} 个",
        f"🔗 裸 URL（非规范）：{len(bare_urls)} 个",
        f"✅ 格式合规率：{compliance_rate:.1f}%",
        "",
        f"🌐 URL 可达性检测（共检测 {len(url_check_results)} 个）：",
        f"   可访问：{reachable_count} / {len(url_check_results)}",
        "",
        "--- URL 逐条检测结果 ---",
    ]
    lines.extend(url_check_results if url_check_results else ["  （报告中未发现任何 URL）"])

    if total_url_count == 0:
        lines.append("\n⚠️ 警告：报告中未检测到任何引用来源，来源可追溯性极差。")
    elif compliance_rate < 60:
        lines.append(f"\n⚠️ 警告：Markdown 格式引用比例偏低（{compliance_rate:.1f}%），建议统一使用 [文字](链接) 格式。")

    return "\n".join(lines)


# ==========================================
# 工具 14：响应时间 / Token 消耗记录器
# ==========================================

# 用模块级变量存储本次运行的统计数据
_run_stats = {
    "start_time": None,
    "llm_call_count": 0,
    "total_input_tokens": 0,
    "total_output_tokens": 0,
}


def start_tracking():
    """在 Research Agent 开始运行前调用，初始化计时器。"""
    _run_stats["start_time"] = time.time()
    _run_stats["llm_call_count"] = 0
    _run_stats["total_input_tokens"] = 0
    _run_stats["total_output_tokens"] = 0


def record_llm_call(response):
    """
    每次 LLM 调用结束后调用此函数，传入 LangChain response 对象，
    自动提取 token 用量并累加。
    """
    _run_stats["llm_call_count"] += 1
    usage = getattr(response, "usage_metadata", None)
    if usage:
        _run_stats["total_input_tokens"] += usage.get("input_tokens", 0)
        _run_stats["total_output_tokens"] += usage.get("output_tokens", 0)


@tool
def get_run_statistics() -> str:
    """
    查询本次 Research Agent 运行的效率统计数据，包括：
    总耗时、LLM 调用次数、输入/输出 Token 用量。

    无需参数，直接调用即可获取当前统计快照。
    """
    if _run_stats["start_time"] is None:
        return "统计尚未启动，请先调用 start_tracking() 初始化。"

    elapsed = time.time() - _run_stats["start_time"]
    total_tokens = _run_stats["total_input_tokens"] + _run_stats["total_output_tokens"]

    lines = [
        "=" * 50,
        "【运行效率统计报告】",
        "=" * 50,
        f"⏱️  总耗时：{elapsed:.1f} 秒",
        f"🤖  LLM 调用次数：{_run_stats['llm_call_count']} 次",
        f"📥  输入 Token：{_run_stats['total_input_tokens']:,}",
        f"📤  输出 Token：{_run_stats['total_output_tokens']:,}",
        f"📊  总 Token 消耗：{total_tokens:,}",
    ]

    # 简单的效率诊断
    if elapsed > 120:
        lines.append("⚠️  耗时较长（>2分钟），建议减少搜索轮次或降低抓取深度。")
    if _run_stats["llm_call_count"] > 20:
        lines.append("⚠️  LLM 调用次数较多，可考虑合并工具调用步骤。")

    return "\n".join(lines)


# ==========================================
# 工具 17：Prompt / 策略优化器
# ==========================================
@tool
def prompt_optimizer(judge_result_json: str, current_prompt: str) -> str:
    """
    根据 LLM-as-Judge 的评分结果，自动诊断薄弱环节并生成改进后的系统 Prompt。
    优化逻辑：覆盖度低 → 增加搜索轮次要求；来源少 → 强制最少引用数；
              深度不足 → 要求对比分析；逻辑混乱 → 强制章节结构。

    Args:
        judge_result_json: llm_as_judge 工具返回的 JSON 字符串。
        current_prompt: 当前 Research Agent 正在使用的系统 Prompt 文本。

    Returns:
        优化后的新系统 Prompt 文本，可直接替换原有 Prompt 使用。
    """
    from agent import critic_llm  # 延迟导入

    # --- Step 1: 解析评分结果，提取薄弱维度 ---
    try:
        judge_data = json.loads(judge_result_json)
        scores = judge_data.get("scores", {})
        weakest = judge_data.get("weakest_dimension", "")
        total_score = judge_data.get("total_score", 0)
        overall_comment = judge_data.get("overall_comment", "")
    except (json.JSONDecodeError, KeyError) as e:
        return f"Prompt 优化失败：无法解析评分 JSON - {str(e)}"

    # --- Step 2: 基于规则生成诊断摘要与优化方向 ---
    rule_hints = []

    coverage_score = scores.get("信息覆盖度", {}).get("score", 5)
    accuracy_score = scores.get("事实准确性", {}).get("score", 5)
    logic_score = scores.get("逻辑连贯性", {}).get("score", 5)
    insight_score = scores.get("深度与洞察", {}).get("score", 5)

    if coverage_score <= 3:
        rule_hints.append(
            "【信息覆盖度不足】要求：至少进行 3 轮搜索，每轮使用不同关键词角度（如背景、原理、应用、对比、最新进展），"
            "每个核心子话题必须有独立的搜索结果支撑。"
        )
    if accuracy_score <= 3:
        rule_hints.append(
            "【事实准确性不足】要求：每项关键数据或结论必须附带原始来源 URL；"
            "对存疑信息需交叉验证至少两个独立来源后方可采用。"
        )
    if logic_score <= 3:
        rule_hints.append(
            "【逻辑连贯性不足】要求：报告必须严格按照以下结构输出：\n"
            "  1. 执行摘要（200字）\n  2. 背景与定义\n  3. 核心内容（各子议题分节）\n"
            "  4. 对比分析表格\n  5. 结论与展望\n  6. 参考文献列表\n"
            "各章节之间需有过渡句，禁止跳跃式切换。"
        )
    if insight_score <= 3:
        rule_hints.append(
            "【深度与洞察不足】要求：报告结论部分必须包含：\n"
            "  (a) 各方案/观点的优劣势对比表格\n"
            "  (b) 基于数据的趋势判断\n"
            "  (c) 至少一条针对特定场景的具体建议\n"
            "禁止在结论部分仅做信息复述。"
        )

    if not rule_hints:
        return f"当前 Prompt 质量良好（总分 {total_score}/20），无需优化。\n\n当前 Prompt：\n{current_prompt}"

    rule_hints_text = "\n\n".join(rule_hints)

    # --- Step 3: 调用 Critic LLM 生成新 Prompt ---
    optimizer_prompt = f"""你是一位 AI Prompt 工程专家。请根据评测诊断结果，对下方系统 Prompt 进行定向优化。

【评测总分】{total_score}/20
【整体评价】{overall_comment}
【最薄弱维度】{weakest}

【规则诊断与优化方向】
{rule_hints_text}

【当前系统 Prompt】
{current_prompt}

---

【任务要求】
1. 在当前 Prompt 的基础上，针对上述诊断结果进行增量修改，不要改变整体调研流程框架。
2. 将诊断出的改进要求自然融入 Prompt 的对应步骤中，语气应像对 Agent 下达明确指令。
3. 直接输出完整的新版 Prompt 文本，不要添加任何说明或前言。
"""

    try:
        response = critic_llm.invoke(optimizer_prompt)
        new_prompt = response.content.strip()

        # 同时将新 Prompt 写入本地文件，供下次运行直接加载
        prompt_path = "optimized_prompt.txt"
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(new_prompt)

        return (
            f"✅ Prompt 优化完成（原总分 {total_score}/20，针对【{weakest}】重点改进）\n"
            f"新 Prompt 已保存至 '{prompt_path}'\n\n"
            f"{'=' * 50}\n【优化后的新系统 Prompt】\n{'=' * 50}\n{new_prompt}"
        )
    except Exception as e:
        return f"Prompt 生成失败：{str(e)}"


# ==========================================
# 统一导出评估工具箱
# ==========================================
eval_tools = [
    llm_as_judge,
    source_traceability_checker,
    get_run_statistics,
    prompt_optimizer,
]