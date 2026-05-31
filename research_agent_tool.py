import os
import re
import time
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from dotenv import load_dotenv
load_dotenv()  # 在文件顶部调用一次就够



# ==========================================
# 1. 网络检索工具 (支持高级语法与时效过滤)
# ==========================================
@tool
def advanced_web_search(query: str, site: str = "", time_range: str = "y") -> str:
    """调用 Tavily API 进行互联网实时检索。支持限定网站范围和时间范围。

    Args:
        query: 检索关键词字符串。
        site: 可选，限定域名（例如 'github.com', 'arxiv.org'），留空则全网搜索。
        time_range: 时间范围限制，默认为 'y'（过去一年内，即精准定位 2025-2026 年最新消息）。可选 'm'(过去一月), 'w'(过去一周)。
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "错误：未配置 TAVILY_API_KEY 环境变量。"

    # 构造请求载荷，告诉搜索引擎要什么
    full_query = query
    if site:
        full_query = f"site:{site} {query}" # 在字符串的开头的引号前加上 f 或 F，即可在字符串内部使用大括号 {} 包裹 Python 表达式或变量。Python 在运行时会自动将大括号内的变量替换为其对应的值。

    url = "https://api.tavily.com/search"
    # payload 请求体 “Tavily API的字典”
    payload = {
        "api_key": api_key,
        "query": full_query,
        "search_depth": "advanced", # 表示使用高级深度搜索，结果更精准，但可能消耗更多 API 额度。
        "include_answer": False, # 不需要对方 AI 生成简答，只需要原始网页结果
        "max_results": 5,
        "time_range": time_range  # 直接透传时效性参数（如过去一年 y），由搜索引擎后端进行时间过滤。
    }

    """
    response 对象内部通常包含以下核心部分：
    response.text：服务器返回的文本内容（字符串格式，如 HTML 或纯文本）。
    response.json()：一个方法。如果服务器返回的是 JSON 格式数据，调用它会将其解析为 Python 字典。
    response.status_code：状态码（如 200 表示成功，404 表示未找到，500 表示服务器错误）。
    response.headers：服务器返回的响应头（包含内容类型、服务器类型、时间等元数据）。
    """

    # 发送网络请求，数据交互
    try:
        response = requests.post(url, json=payload, timeout=10) # 通过 HTTP POST 方法，将 payload 以 JSON 格式发送到指定的 url。设置 timeout=10 防止网络卡死时程序无限期等待。
        if response.status_code != 200: #status_code HTTP校验码 200代表请求成功
            return f"搜索请求失败，状态码: {response.status_code}"

        # 解析与数据清洗
        results = response.json().get("results", []) # .get("results", []),提取键名为results的数据否则为空列表
        if not results:
            return f"未找到关于 '{full_query}' 的相关结果。"

        formatted_results = [] #只提取title，url，content
        for item in results:
            formatted_results.append(
                f"标题: {item['title']}\n链接: {item['url']}\n摘要: {item['content']}\n"
            )
        return "\n---\n".join(formatted_results) #将列表（或可迭代对象）formatted_results 中的所有字符串元素连接成一个单一的字符串，并使用 \n---\n作为分隔符插在每个元素之间
    except Exception as e:
        return f"搜索发生异常: {str(e)}"


# ==========================================
# 2. 网页内容深度抓取工具 (带抗噪清洗与安全分块) 将任意网页的干扰信息（广告、导航栏）剔除，提取纯净的正文，并将其切分成固定大小的“数据块”（Chunks），以防止大模型因单词数过多而“撑爆”内存（上下文溢出）。
# ==========================================
@tool
def smart_web_scraper(url: str, chunk_index: int = 0) -> str:
    """接收指定 URL，自动绕过广告、导航栏并提取核心正文。支持长文本分块读取，防止上下文溢出。

    Args:
        url: 网页的完整 URL 字符串。
        chunk_index: 当网页文本过长被切片时，指定读取第几块内容（每块 2500 字）。默认为 0（第一块）。
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = response.apparent_encoding # 自动检测网页是用 UTF-8 还是 GBK 编码，防止抓下来的中文变成乱码。

        if response.status_code != 200:
            return f"抓取失败，HTTP 状态码: {response.status_code}"

        soup = BeautifulSoup(response.text, 'html.parser') # 将文本和解析器结合，在内存中生成一个名为 soup 的“结构化 DOM 树”对象。

        # 强力剥离无用干扰标签
        for noise in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe"]):
            noise.extract() # 剔除网页里的 JS 脚本（script）、样式表（style）、导航栏（nav）、页脚（footer）、侧边栏（aside）

        # 提取并清洗文本结构 将文本拆分成多行，去除每行两端的空白字符（空格、制表符、换行符），并自动过滤掉所有空行。
        text = soup.get_text(separator='\n') #get_text提取纯文字，separator=’\n‘ 当两个标签的文字挨在一起时，指定用换行符 \n 把它们隔开
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        # text.splitlines()：把整段文本按照行切开，变成一个“一行行文字”的列表。
        # if line.strip()如果该行只有空格或原本就是空字符串，line.strip() 会返回空字符串 ""（在布尔判断中为 False），该行即被丢弃。只有包含实际内容的行（True）才会进入下一步。
        # line.strip()通过筛选的再次执行，切除空格和换行符
        clean_text = "\n".join(lines) # 连接成字符串

        # 核心优化：文本安全分块 (Chunking) 策略
        chunk_size = 2500
        chunks = [clean_text[i:i + chunk_size] for i in range(0, len(clean_text), chunk_size)]

        if not chunks:
            return "该网页未提取到有效文本正文。"

        total_chunks = len(chunks)
        if chunk_index >= total_chunks:
            return f"错误：指定的块索引 {chunk_index} 超出范围。总块数: {total_chunks}"

        return f"[分块日志] 当前读取第 {chunk_index + 1}/{total_chunks} 块内容：\n\n{chunks[chunk_index]}"
    except Exception as e:
        return f"抓取发生异常: {str(e)}"


# ==========================================
# 3. 学术文献检索工具 (对接 arXiv 官方开放接口)
# ==========================================
@tool
def academic_literature_search(query: str, max_results: int = 3) -> str:
    """对接 arXiv 学术数据库，专门用于检索计算机科学、大模型技术原理、算法对比等深度调研论文。
    返回论文的标题、作者、发布日期、摘要及 PDF 链接。

    Args:
        query: 学术检索关键词（英文关键词效果最佳，例如 'Transformer LLM inference optimization'）。
        max_results: 返回的高质量文献数量，默认 3 篇。
    """
    # 格式化检索词，arXiv 要求空格替换为加号
    formatted_query = "+".join(query.split())
    # 默认按发布时间降序排序 (sortBy=submittedDate, sortOrder=descending) 确保获取 2025-2026 最新论文
    url = f"http://export.arxiv.org/api/query?search_query=all:{formatted_query}&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"

    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return f"学术接口连接失败，状态码: {response.status_code}"

        soup = BeautifulSoup(response.text, 'xml')  # 使用 xml 解析器解析 arXiv 返回的 atom 订阅流
        entries = soup.find_all('entry') # 在 arXiv 返回的 XML 数据中，每一篇独立的论文都被包裹在一个个 <entry> 标签对里
        #f ind_all()把所有检索到的论文打包成一个列表
        if not entries:
            return f"arXiv 数据库中未检索到与 '{query}' 相关的最新文献。"

        articles = []
        for entry in entries:
            title = entry.title.text.strip().replace('\n', ' ')
            summary = entry.summary.text.strip().replace('\n', ' ')
            published = entry.published.text[:10]  # 提取 YYYY-MM-DD
            pdf_link = ""
            for link in entry.find_all('link'):
                if link.get('title') == 'pdf':
                    pdf_link = link.get('href')

            articles.append(
                f"【论文标题】: {title}\n【发布日期】: {published}\n【核心摘要】: {summary}\n【PDF链接】: {pdf_link}\n"
            )
        return "\n---\n".join(articles)
    except Exception as e:
        return f"学术检索发生异常: {str(e)}"


# ==========================================
# 4. 本地文件读写工具 (File I/O 管理器)
# ==========================================
@tool
def local_file_manager(action: str, file_name: str, content: str = "") -> str:
    """本地调研文件的持久化读写工具。支持创建、追加或读取调研过程中的中间暂存数据和最终 Markdown 报告。

    Args:
        action: 必须是 'write'(覆盖写入)、'append'(末尾追加) 或 'read'(读取文件) 之一。
        file_name: 本地文件名（例如 'research_draft.txt' 或 'final_report.md'）。
        content: 只有在 action 为 'write' 或 'append' 时需要传入的文本内容。
    """
    # 安全沙箱路径限制：只允许在当前工作目录及其子目录下读写，防止越权
    safe_path = os.path.basename(file_name)

    try:
        if action == "write":
            with open(safe_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"成功：内容已成功写入本地文件 '{safe_path}'。"

        elif action == "append":
            with open(safe_path, "a", encoding="utf-8") as f:  # 用 a 模式
                f.write("\n" + content)
            return f"成功：内容已成功追加至本地文件 '{safe_path}' 末尾。"

        elif action == "read":
            if not os.path.exists(safe_path):
                return f"错误：本地未找到文件 '{safe_path}'。"
            with open(safe_path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            return "错误：action 参数必须是 'write'、'append' 或 'read'。"
    except Exception as e:
        return f"文件操作失败，异常原因: {str(e)}"


# ==========================================
# 统一导出完整工具箱
# ==========================================
research_tools = [
    advanced_web_search,
    smart_web_scraper,
    academic_literature_search,
    local_file_manager
]