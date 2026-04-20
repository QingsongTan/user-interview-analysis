"""
用户访谈自动化分析工具 — AI增强的用研工作流
可视化界面版（Gradio）
"""

import os
import gradio as gr
import html
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

# ========== LLM 配置 ==========
client = None
client_api_key = None
MODEL = "qwen-plus"
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
_last_report_data: dict = {}


def get_backend_client() -> OpenAI:
    global client, client_api_key
    api_key = DASHSCOPE_API_KEY.strip()
    if not api_key:
        raise gr.Error("后端未配置 DASHSCOPE_API_KEY，请先在 interview_analysis_app.py 顶部填写阿里云百炼 API Key。")
    if client is None or client_api_key != api_key:
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        client_api_key = api_key
    return client


def call_llm(system_prompt: str, user_prompt: str) -> str:
    response = get_backend_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=4096,
        temperature=0.3
    )
    return response.choices[0].message.content


def escape_html(value) -> str:
    return html.escape("" if value is None else str(value))


def build_empty_panel(title: str, description: str, note: str = "") -> str:
    note_html = f'<div class="empty-panel__note">{escape_html(note)}</div>' if note else ""
    return f"""
<div class="empty-panel">
  <div class="empty-panel__eyebrow">待填充</div>
  <div class="empty-panel__title">{escape_html(title)}</div>
  <div class="empty-panel__desc">{escape_html(description)}</div>
  {note_html}
</div>
"""


WORKFLOW_STEPS = ["文本预处理", "主题编码", "情感分析", "亲和图聚类", "洞察生成"]
WORKFLOW_STAGE_INDEX = {
    "文本预处理完成": 0,
    "主题编码完成": 1,
    "主题聚类完成": 3,
    "分析完成": 4,
}

DETAIL_MODULES = [
    {
        "number": "01",
        "title": "主题编码",
        "anchor": "detail-codes",
        "hint": "按问题展示主题标签、原文引述和研究员备注。",
    },
    {
        "number": "02",
        "title": "情感分析",
        "anchor": "detail-sentiment",
        "hint": "展示每段访谈的整体情感和细分方面。",
    },
    {
        "number": "03",
        "title": "亲和图聚类",
        "anchor": "detail-affinity",
        "hint": "汇总多个主题标签，形成分组卡片。",
    },
    {
        "number": "04",
        "title": "关键洞察",
        "anchor": "detail-insights",
        "hint": "沉淀可执行的研究发现与后续追问方向。",
    },
]


def render_section_header(title: str, description: str = "", kicker: str = "") -> str:
    kicker_html = f'<div class="section-eyebrow">{escape_html(kicker)}</div>' if kicker else ""
    desc_html = f'<div class="section-desc">{escape_html(description)}</div>' if description else ""
    return f"""
<div class="section-header">
  {kicker_html}
  <div class="section-title">{escape_html(title)}</div>
  {desc_html}
</div>
"""


def build_workflow_strip(active_index: int | None = None) -> str:
    chips = []
    for index, label in enumerate(WORKFLOW_STEPS):
        if active_index is None:
            state = "pending"
        elif index < active_index:
            state = "done"
        elif index == active_index:
            state = "active"
        else:
            state = "pending"
        chips.append(
            f'<span class="workflow-chip workflow-chip--{state}">{escape_html(label)}</span>'
        )
    return f'<div class="workflow-strip" aria-label="分析流程">{"".join(chips)}</div>'


def build_detail_directory() -> str:
    chips = []
    for module in DETAIL_MODULES:
        chips.append(
            f'<a class="detail-nav__chip" href="#{module["anchor"]}">'
            f'<span class="detail-nav__chip-index">{escape_html(module["number"])}</span>'
            f'<span class="detail-nav__chip-title">{escape_html(module["title"])}</span>'
            "</a>"
        )
    return f"""
<div class="detail-nav" aria-label="结果明细目录">
  <div class="detail-nav__head">
    <div class="detail-nav__eyebrow">目录</div>
    <div class="detail-nav__title">结果明细导航</div>
  </div>
  <div class="detail-nav__chips">
    {"".join(chips)}
  </div>
  <div class="detail-nav__hint">点击目录跳转到对应模块，再按需展开查看细节。长访谈时可以先扫总览，再只打开需要的部分。</div>
</div>
"""


def build_summary_placeholder() -> str:
    return """
<div class="summary-panel summary-panel--empty">
  <div class="empty-panel">
    <div class="empty-panel__eyebrow">结果总览</div>
    <div class="empty-panel__title">等待分析</div>
    <div class="empty-panel__desc">在左侧完成配置并粘贴访谈文本后，这里会按流程逐步回填结果。</div>
    <div class="empty-panel__note">分析路径会依次经过文本预处理、主题编码、情感分析、亲和图聚类与洞察生成。</div>
    <div class="workflow-strip" aria-label="分析流程">
      <span class="workflow-chip workflow-chip--pending">文本预处理</span>
      <span class="workflow-chip workflow-chip--pending">主题编码</span>
      <span class="workflow-chip workflow-chip--pending">情感分析</span>
      <span class="workflow-chip workflow-chip--pending">亲和图聚类</span>
      <span class="workflow-chip workflow-chip--pending">洞察生成</span>
    </div>
  </div>
</div>
"""


def build_summary_html(
    segments_count: int,
    total_themes: int,
    total_clusters: int,
    sentiments_count: int,
    model_name: str,
    stage_label: str,
) -> str:
    active_index = WORKFLOW_STAGE_INDEX.get(stage_label)
    rows = [
        ("访谈片段", segments_count, "自动拆分后的有效问答段"),
        ("主题标签", total_themes, "提取到的原文主题编码"),
        ("聚类分组", total_clusters, "亲和图归纳后的分组数"),
        ("情感分析", sentiments_count, "完成情感标注的片段数"),
    ]
    ledger_rows = "".join(
        f"""
        <div class="summary-row">
          <div class="summary-row__meta">
            <div class="summary-row__label">{escape_html(label)}</div>
            <div class="summary-row__hint">{escape_html(hint)}</div>
          </div>
          <div class="summary-row__value">{escape_html(value)}</div>
        </div>
        """
        for label, value, hint in rows
    )
    return f"""
<div class="summary-panel">
  <div class="summary-head">
    <div>
      <div class="section-eyebrow">{escape_html(stage_label)}</div>
      <div class="section-title">研究摘要</div>
      <div class="section-desc">先看总览，再向下查看编码、情感、聚类和洞察。所有结论都保留原文证据，便于回溯。</div>
    </div>
    <div class="summary-badge">{escape_html(model_name)}</div>
  </div>
  <div class="summary-copy">结果会随着分析进度分段回填，方便你在任一阶段停下来核对证据。</div>
  <div class="summary-ledger">
    {ledger_rows}
  </div>
  {build_workflow_strip(active_index)}
</div>
"""


# ========== 分析模块 ==========

_QA_PATTERN = re.compile(r'(Q\d+)[：:](.+?)\n\s*(?:P\d+|受访者)[：:](.+?)(?=\nQ\d+[：:]|\Z)', re.DOTALL)
_JSON_ARRAY_PATTERN = re.compile(r'\[.*\]', re.DOTALL)
_JSON_OBJECT_PATTERN = re.compile(r'\{.*\}', re.DOTALL)


def preprocess_transcript(raw_text: str) -> list[dict]:
    """将访谈文本拆分为 Q&A 对"""
    matches = _QA_PATTERN.findall(raw_text)
    if not matches:
        # 降级：按段落分割
        paragraphs = [p.strip() for p in raw_text.split('\n\n') if p.strip() and len(p.strip()) > 20]
        return [{"id": f"P{i+1}", "question": "（自动分段）", "answer": p} for i, p in enumerate(paragraphs)]
    return [{"id": qid, "question": q.strip(), "answer": a.strip()} for qid, q, a in matches]


def code_themes(segments: list[dict], progress=None) -> dict:
    system = """你是一位专业的用户研究员，擅长对用户访谈文本进行主题编码。
请严格按照以下规则进行编码：
1. 每段回答可能包含多个主题，请逐一提取
2. 每个主题必须配对原文引述（直接引用受访者原话）
3. 主题标签要简洁、一致，便于后续聚类
4. 输出严格的 JSON 格式"""

    def _code_one(seg: dict) -> tuple[str, list]:
        user_prompt = f"""请对以下访谈片段进行主题编码。

问题：{seg['question']}
回答：{seg['answer']}

请输出JSON数组，每个元素包含：
- "theme": 主题标签（2-6个字）
- "quote": 原文引述（直接摘录受访者原话）
- "note": 研究员备注（一句话解读）

仅输出JSON，不要其他文字。"""
        result = call_llm(system, user_prompt)
        json_match = _JSON_ARRAY_PATTERN.search(result)
        return seg['id'], json.loads(json_match.group()) if json_match else []

    all_codes = {}
    completed = 0
    max_workers = min(len(segments), 5)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {executor.submit(_code_one, seg): seg['id'] for seg in segments}
        for future in as_completed(future_to_id):
            seg_id, codes = future.result()
            all_codes[seg_id] = codes
            completed += 1
            if progress:
                progress(
                    0.1 + completed / len(segments) * 0.3,
                    desc=f"主题编码 {completed}/{len(segments)} 完成",
                )
    # 按原始顺序排列
    ordered = {seg['id']: all_codes.get(seg['id'], []) for seg in segments}
    return ordered


def analyze_sentiment(segments: list[dict]) -> list[dict]:
    system = """你是用户研究情感分析专家。
请对每段用户回答进行情感标注。
注意区分"对产品功能的情感"和"对体验问题的情感"，同一段回答中可能包含多种情感。"""

    text_block = "\n\n".join(f"[{s['id']}] {s['answer']}" for s in segments)
    user_prompt = f"""请对以下各段访谈回答进行情感分析。

{text_block}

输出JSON数组，每个元素包含：
- "id": 片段编号
- "overall": 整体情感（"正面"/"负面"/"中性"/"混合"）
- "details": 数组，每个子项包含 {{"aspect": "方面", "sentiment": "正面/负面/中性", "intensity": 1-5, "evidence": "原文依据"}}

仅输出JSON。"""
    result = call_llm(system, user_prompt)
    json_match = _JSON_ARRAY_PATTERN.search(result)
    return json.loads(json_match.group()) if json_match else []


def cluster_themes(all_codes: dict) -> dict:
    all_themes = []
    for qid, codes in all_codes.items():
        for c in codes:
            all_themes.append({"source": qid, "theme": c['theme'], "quote": c['quote']})

    themes_text = json.dumps(all_themes, ensure_ascii=False, indent=2)
    user_prompt = f"""以下是从一场用户访谈中提取的所有主题标签和引述：

{themes_text}

请将这些主题进行聚类分组（类似亲和图/Affinity Diagram），输出JSON：
{{
  "clusters": [
    {{
      "group_name": "分组名称",
      "description": "该分组的核心含义",
      "themes": ["包含的主题标签列表"],
      "representative_quote": "最能代表这个分组的一条引述"
    }}
  ]
}}

分组数量控制在3-6个，仅输出JSON。"""
    system = "你是用户研究分析专家，擅长使用亲和图法（Affinity Diagram）对定性数据进行归纳分组。"
    result = call_llm(system, user_prompt)
    json_match = _JSON_OBJECT_PATTERN.search(result)
    return json.loads(json_match.group()) if json_match else {}


def generate_insights(affinity: dict, sentiments: list) -> str:
    context = json.dumps({
        "affinity_clusters": affinity,
        "sentiment_analysis": sentiments
    }, ensure_ascii=False, indent=2)

    user_prompt = f"""基于以下访谈分析结果，请生成研究洞察报告。

分析数据：
{context}

请按以下格式输出（Markdown）：

### 核心发现
（3-5条关键洞察，每条包含：发现 + 证据引述 + 对产品的启示）

### 用户需求优先级
（按紧迫度排序的需求列表）

### 值得深入探索的方向
（基于本次访谈发现的、值得后续研究的问题）

注意：每个发现都必须有原文引述作为支撑，不得臆测。"""
    system = "你是腾讯用户研究团队的高级研究员，正在撰写QQ产品的用户访谈分析报告。请输出专业、有洞见、可执行的研究发现。"
    return call_llm(system, user_prompt)


# ========== 格式化输出 ==========

def format_codes_html(all_codes: dict) -> str:
    sections = []
    total = 0
    for qid, codes in all_codes.items():
        rows = []
        for c in codes:
            theme = escape_html(c.get("theme", ""))
            quote = escape_html(c.get("quote", ""))
            note = escape_html(c.get("note", ""))
            rows.append(
                f"""
                <div class="data-row">
                  <div class="data-cell data-cell--strong">{theme}</div>
                  <div class="data-cell data-cell--quote">“{quote}”</div>
                  <div class="data-cell">{note}</div>
                </div>
                """
            )
            total += 1
        body = "".join(rows) if rows else '<div class="empty-inline">当前问题尚未提取到主题编码。</div>'
        sections.append(
            f"""
            <section class="report-section">
              <div class="report-section-head">
                <div>
                  <div class="report-kicker">{escape_html(qid)}</div>
                  <h4 class="report-title">主题编码</h4>
                </div>
                <div class="report-count">{len(codes)} 条</div>
              </div>
              <div class="data-table">
                <div class="data-row data-row--head">
                  <div>主题标签</div>
                  <div>原文引述</div>
                  <div>研究员备注</div>
                </div>
                {body}
              </div>
            </section>
            """
        )
    if not sections:
        return build_empty_panel("主题编码", "当前没有可展示的编码结果。", "请先运行一次分析。")
    return f"""
<div class="report-stack">
  <div class="report-summary">共提取 {total} 个主题标签</div>
  {''.join(sections)}
</div>
"""


def format_sentiment_html(sentiments: list) -> str:
    badge_class_map = {
        "正面": "badge--positive",
        "负面": "badge--negative",
        "中性": "badge--neutral",
        "混合": "badge--mixed",
    }
    sections = []
    for s in sentiments:
        overall = s.get("overall", "未知")
        details = []
        for d in s.get("details", []):
            intensity = d.get("intensity", 3)
            try:
                intensity = max(1, min(int(intensity), 5))
            except (TypeError, ValueError):
                intensity = 3
            bar = "█" * intensity + "░" * (5 - intensity)
            detail_badge = badge_class_map.get(d.get("sentiment", ""), "badge--unknown")
            details.append(
                f"""
                <div class="detail-row">
                  <div class="detail-cell detail-cell--strong">{escape_html(d.get("aspect", ""))}</div>
                  <div class="detail-cell"><span class="badge {detail_badge}">{escape_html(d.get("sentiment", ""))}</span></div>
                  <div class="detail-cell"><code class="intensity-pill">{escape_html(bar)}</code></div>
                  <div class="detail-cell detail-cell--quote">“{escape_html(d.get("evidence", ""))}”</div>
                </div>
                """
            )
        sections.append(
            f"""
            <section class="report-section">
              <div class="report-section-head">
                <div>
                  <div class="report-kicker">{escape_html(s.get("id", "?"))}</div>
                  <h4 class="report-title">情感分析</h4>
                </div>
                <div class="badge {badge_class_map.get(overall, 'badge--unknown')}">{escape_html(overall)}</div>
              </div>
              <div class="detail-table">
                <div class="detail-row detail-row--head">
                  <div>方面</div>
                  <div>情感</div>
                  <div>强度</div>
                  <div>原文依据</div>
                </div>
                {''.join(details) if details else '<div class="empty-inline">当前片段尚未生成情感细分。</div>'}
              </div>
            </section>
            """
        )
    if not sections:
        return build_empty_panel("情感分析", "当前没有可展示的情感结果。", "请先运行一次分析。")
    return f"""
<div class="report-stack">
  {''.join(sections)}
</div>
"""


def format_affinity_html(affinity: dict) -> str:
    clusters = affinity.get("clusters", [])
    cards = []
    for idx, cluster in enumerate(clusters, start=1):
        themes = cluster.get("themes", [])
        theme_tags = "".join(f'<span class="chip">{escape_html(t)}</span>' for t in themes)
        if not theme_tags:
            theme_tags = '<span class="chip chip--empty">暂无主题</span>'
        cards.append(
            f"""
            <article class="cluster-card">
              <div class="report-kicker">分组 {idx:02d}</div>
              <h4 class="report-title">{escape_html(cluster.get("group_name", "未命名"))}</h4>
              <p class="cluster-desc">{escape_html(cluster.get("description", ""))}</p>
              <div class="chip-wrap">{theme_tags}</div>
              <blockquote class="quote-box">{escape_html(cluster.get("representative_quote", ""))}</blockquote>
            </article>
            """
        )
    if not cards:
        return build_empty_panel("亲和图聚类", "当前没有可展示的聚类结果。", "请先运行一次分析。")
    return f"""
<div class="report-stack">
  <div class="cluster-grid">
    {''.join(cards)}
  </div>
</div>
"""


# ========== 主分析流程 ==========

def run_analysis(transcript: str, model_choice: str, progress=gr.Progress()):
    """一键执行完整分析流程"""
    global MODEL

    if not transcript.strip():
        raise gr.Error("请输入访谈文本！")

    # 后端密钥只从脚本顶部常量读取，前端不暴露
    get_backend_client()
    MODEL = model_choice

    # Step 1: 预处理
    segments = preprocess_transcript(transcript)
    if not segments:
        raise gr.Error("未能从文本中提取到有效的问答对，请检查格式。")

    progress(0.05, desc="文本预处理完成，准备主题编码")

    yield (
        build_summary_html(
            segments_count=len(segments),
            total_themes=0,
            total_clusters=0,
            sentiments_count=0,
            model_name=MODEL,
            stage_label="文本预处理完成",
        ),
        build_empty_panel(
            "主题编码",
            "文本已拆分完成，正在开始主题编码。",
            "完成后会按问题展示主题标签、原文引述和研究员备注。",
        ),
        build_empty_panel(
            "情感分析",
            "等待主题编码阶段完成后开始情感分析。",
            "这里会按片段展示整体情感和细分方面。",
        ),
        build_empty_panel(
            "亲和图聚类",
            "等待前序分析完成后生成聚类结果。",
            "这里会汇总多个主题标签，形成分组卡片。",
        ),
        "*洞察生成中，当前先完成结构化拆分。*",
    )

    # Step 2: 主题编码
    progress(0.1, desc="开始主题编码")
    all_codes = code_themes(segments, progress=progress)

    codes_html = format_codes_html(all_codes)
    total_themes = sum(len(v) for v in all_codes.values())

    progress(0.4, desc="主题编码完成，准备情感分析")

    yield (
        build_summary_html(
            segments_count=len(segments),
            total_themes=total_themes,
            total_clusters=0,
            sentiments_count=0,
            model_name=MODEL,
            stage_label="主题编码完成",
        ),
        codes_html,
        build_empty_panel(
            "情感分析",
            "主题编码已完成，正在进入情感分析。",
            "这里会显示每段访谈的整体情感和细分方面。",
        ),
        build_empty_panel(
            "亲和图聚类",
            "等待情感分析完成后生成聚类结果。",
            "这里会汇总多个主题标签，形成分组卡片。",
        ),
        "*情感分析和后续洞察生成中...*",
    )

    # Step 3 + 4: 情感分析与主题聚类并行执行（互不依赖）
    progress(0.45, desc="情感分析与主题聚类并行执行中")
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_sentiment = executor.submit(analyze_sentiment, segments)
        future_affinity = executor.submit(cluster_themes, all_codes)
        sentiments = future_sentiment.result()
        affinity = future_affinity.result()

    sentiment_html = format_sentiment_html(sentiments)
    affinity_html = format_affinity_html(affinity)
    total_clusters = len(affinity.get("clusters", []))

    progress(0.65, desc="主题聚类完成，整理洞察")

    yield (
        build_summary_html(
            segments_count=len(segments),
            total_themes=total_themes,
            total_clusters=total_clusters,
            sentiments_count=len(sentiments),
            model_name=MODEL,
            stage_label="主题聚类完成",
        ),
        codes_html,
        sentiment_html,
        affinity_html,
        "*洞察生成中，正在整理研究发现。*",
    )

    # Step 5: 洞察生成
    progress(0.8, desc="生成关键洞察")
    insights = generate_insights(affinity, sentiments)

    progress(0.95, desc="洞察整理完成")
    summary = build_summary_html(
        segments_count=len(segments),
        total_themes=total_themes,
        total_clusters=total_clusters,
        sentiments_count=len(sentiments),
        model_name=MODEL,
        stage_label="分析完成",
    )

    progress(1.0, desc="分析完成")
    yield summary, codes_html, sentiment_html, affinity_html, insights


# ========== 示例数据 ==========

DEMO_TRANSCRIPT = """【受访者信息】
编号：U02，女，28岁，白领上班族，淘宝购物经验5年，月均使用闪购2-3次

【访谈内容】

Q1：能说说你平时怎么使用淘宝闪购的吗？
U02：闪购对我来说就是个"猎人游戏"。我经常刷刷看有没有我想要的东西突然降价。特别是护肤品和服装，上次买过一个颈部按摩仪，闪购价比正价便宜了一半，那一刻真的很兴奋。我还加入了几个闪购的推送群，有朋友一发现好货就在群里喊，大家一起冲。有时候错过了也挺遗憾的，下次就会更加留意。

Q2：闪购最吸引你的地方是什么？
U02：肯定是价格啊。有些品牌的东西平时舍不得买，但闪购的时候价格就特别诱人。我算过，闪购至少能便宜30-50%。而且有时候闪购会推出一些日常找不到的款，算是打捡漏。还有一个好处是时间限制让我有紧迫感，我平时容易纠结，闪购的"限时限量"反而帮我快速决策。

Q3：使用过程中有没有不满意的地方？
U02：有啊，首先是通知太不稳定了。我明明开了推送，但经常看不到，等我发现的时候已经没货了。前两天就这样，错过了一个大牌美妆的闪购，特别遗憾。还有就是搜索功能太弱，我想找特定的品牌或品类的闪购历史记录特别难，只能一个一个翻。另外闪购的商品质量良莠不齐，有时候页面写得很吸引，但实际收到的东西不太理想，退货也很麻烦。

Q4：你会同时在京东、拼多多这些平台看限时活动吗？
U02：会啊。说实话，现在京东的"秒杀"活动我也经常看，拼多多的百亿补贴性价比有时候比淘宝闪购还高。但淘宝闪购的品类更齐全，特别是大牌化妆品和衣服，闪购的选择更多。京东的秒杀总感觉没那么容易买到，总是显示"已抢光"。拼多多虽然便宜但我有点担心品质。所以我还是主要在淘宝闪购，偶尔才用其他平台。

Q5：你觉得闪购应该怎么改进？
U02：首先推送一定要可靠，这是基础。能不能有个"我的关注"功能，让我关注特定的品牌或品类，只要这些品牌有闪购就通知我？还有就是搜索和浏览历史要好一点，有时候我看过的东西想再看一次都找不到。另外，能不能在闪购前给个"预告"？有时候我在等某个大促，提前知道会更有计划。最后就是退货流程要简化，现在退货太麻烦了。

Q6：你身边的朋友对淘宝闪购的态度怎么样？
U02：我们公司有个"淘宝闪购"微信群，有十几个人，每天都有人在里面分享好货。说明大家都比较关注。但我也有朋友根本不用闪购，他们说太费时间了，"限时限量"让他们有心理压力，宁可多花点钱也不想受这种压力。还有些朋友用过一两次但后来就没怎么用了，说是经常错过，或者买回来后悔。我觉得闪购这种模式确实不是所有人都适应。"""


# ========== Gradio 界面 ==========

CUSTOM_CSS = """
:root {
    --surface: rgba(255, 255, 255, 0.86);
    --surface-strong: rgba(255, 255, 255, 0.96);
    --border: rgba(15, 23, 42, 0.08);
    --border-strong: rgba(37, 99, 235, 0.16);
    --text-strong: #0f172a;
    --text: #334155;
    --muted: #64748b;
    --primary: #2563eb;
    --primary-soft: rgba(37, 99, 235, 0.10);
    --primary-soft-2: rgba(37, 99, 235, 0.06);
    --shadow: 0 24px 60px rgba(15, 23, 42, 0.08);
    --radius-xl: 28px;
    --radius-lg: 20px;
    --radius-md: 16px;
}

.gradio-container {
    background:
        radial-gradient(circle at top left, rgba(37, 99, 235, 0.16), transparent 28%),
        radial-gradient(circle at top right, rgba(15, 23, 42, 0.08), transparent 24%),
        linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
    color: var(--text);
}

body {
    font-family: Inter, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
}

.app-shell {
    width: min(100%, 1440px);
    margin: 0 auto;
    padding: 24px 18px 40px;
    gap: 18px;
}

.hero-card {
    display: flex;
    align-items: stretch;
    justify-content: space-between;
    gap: 24px;
    padding: 28px 30px;
    border-radius: 30px;
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.96) 0%, rgba(30, 64, 175, 0.90) 100%);
    color: white;
    box-shadow: 0 26px 60px rgba(15, 23, 42, 0.18);
    border: 1px solid rgba(255, 255, 255, 0.08);
}

.hero-copy {
    flex: 1 1 auto;
    min-width: 0;
}

.hero-eyebrow {
    margin: 0 0 10px;
    font-size: 0.75rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(191, 219, 254, 0.85);
    font-weight: 700;
}

.hero-title {
    margin: 0;
    font-size: clamp(1.8rem, 3vw, 2.6rem);
    line-height: 1.1;
    font-weight: 800;
    color: white;
}

.hero-lead {
    margin: 14px 0 0;
    max-width: 66ch;
    font-size: 1rem;
    line-height: 1.8;
    color: rgba(226, 232, 240, 0.92);
}

.hero-meta {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
    min-width: 280px;
    align-content: start;
}

.meta-chip {
    padding: 12px 14px;
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid rgba(255, 255, 255, 0.14);
    color: white;
    font-size: 0.92rem;
    line-height: 1.4;
}

.workspace-grid {
    gap: 18px;
    align-items: flex-start;
}

.panel-card {
    background: var(--surface);
    backdrop-filter: blur(18px);
    border: 1px solid var(--border);
    border-radius: var(--radius-xl);
    box-shadow: var(--shadow);
    padding: 20px;
}

.panel-card + .panel-card {
    margin-top: 18px;
}

.section-eyebrow {
    margin: 0 0 6px;
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--primary);
}

.section-title {
    margin: 0;
    font-size: 1.12rem;
    font-weight: 800;
    color: var(--text-strong);
    line-height: 1.3;
}

.section-desc {
    margin: 8px 0 16px;
    color: var(--muted);
    font-size: 0.94rem;
    line-height: 1.7;
}

.action-row {
    margin-top: 10px;
}

.action-row .gr-button {
    min-height: 48px;
    border-radius: 14px;
}

.action-row .gr-button-primary {
    box-shadow: 0 10px 24px rgba(37, 99, 235, 0.20);
}

.summary-panel {
    display: flex;
    flex-direction: column;
    gap: 16px;
}

.summary-head {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 16px;
}

.summary-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-height: 40px;
    padding: 0 14px;
    border-radius: 999px;
    background: var(--primary-soft);
    color: var(--primary);
    border: 1px solid var(--border-strong);
    font-size: 0.84rem;
    font-weight: 700;
    white-space: nowrap;
}

.summary-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
}

.summary-stat {
    padding: 14px 16px;
    border-radius: 18px;
    background: linear-gradient(180deg, rgba(37, 99, 235, 0.10), rgba(255, 255, 255, 0.94));
    border: 1px solid rgba(37, 99, 235, 0.12);
}

.summary-stat__label {
    font-size: 0.78rem;
    letter-spacing: 0.05em;
    color: var(--muted);
    margin-bottom: 6px;
}

.summary-stat__value {
    font-size: 1.8rem;
    font-weight: 800;
    color: var(--text-strong);
    line-height: 1.1;
}

.summary-stat__hint {
    margin-top: 6px;
    font-size: 0.84rem;
    line-height: 1.5;
    color: var(--muted);
}

.workflow-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.workflow-chip,
.chip {
    display: inline-flex;
    align-items: center;
    padding: 7px 11px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    line-height: 1;
}

.workflow-chip {
    background: rgba(37, 99, 235, 0.08);
    color: #1d4ed8;
}

.chip {
    background: #eff6ff;
    color: #1d4ed8;
    border: 1px solid rgba(37, 99, 235, 0.08);
}

.chip--empty {
    background: rgba(148, 163, 184, 0.16);
    color: #475569;
}

.report-stack {
    display: flex;
    flex-direction: column;
    gap: 16px;
}

.report-summary {
    padding: 14px 16px;
    border-radius: 18px;
    background: linear-gradient(180deg, rgba(37, 99, 235, 0.08), rgba(255, 255, 255, 0.94));
    border: 1px solid rgba(37, 99, 235, 0.12);
    color: var(--text);
    font-size: 0.94rem;
    line-height: 1.6;
}

.report-section,
.cluster-card,
.empty-panel {
    background: var(--surface-strong);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
}

.report-section {
    display: flex;
    flex-direction: column;
    gap: 12px;
    padding: 18px;
}

.report-section-head {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 12px;
}

.report-kicker {
    margin: 0 0 4px;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--primary);
    font-weight: 800;
}

.report-title {
    margin: 0;
    font-size: 1.02rem;
    font-weight: 800;
    color: var(--text-strong);
}

.report-count {
    display: inline-flex;
    align-items: center;
    min-height: 34px;
    padding: 0 12px;
    border-radius: 999px;
    background: var(--primary-soft-2);
    color: var(--primary);
    font-size: 0.82rem;
    font-weight: 700;
    white-space: nowrap;
}

.data-table,
.detail-table {
    display: flex;
    flex-direction: column;
    gap: 1px;
    background: rgba(148, 163, 184, 0.16);
    border-radius: 16px;
    overflow: hidden;
}

.data-row,
.detail-row {
    display: grid;
    gap: 1px;
    background: rgba(255, 255, 255, 0.92);
}

.data-row {
    grid-template-columns: 1.1fr 1.8fr 1.2fr;
}

.detail-row {
    grid-template-columns: 1.1fr 0.9fr 0.7fr 2fr;
}

.data-row > div,
.detail-row > div {
    padding: 12px 14px;
    font-size: 0.92rem;
    line-height: 1.65;
    color: var(--text);
    min-width: 0;
}

.data-row--head,
.detail-row--head {
    background: rgba(241, 245, 249, 0.96);
    color: #64748b;
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.data-cell--strong,
.detail-cell--strong {
    font-weight: 800;
    color: var(--text-strong);
}

.data-cell--quote,
.detail-cell--quote {
    color: #1e293b;
}

.badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-height: 30px;
    padding: 0 10px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    white-space: nowrap;
}

.badge--positive {
    background: rgba(34, 197, 94, 0.12);
    color: #166534;
}

.badge--negative {
    background: rgba(239, 68, 68, 0.12);
    color: #991b1b;
}

.badge--neutral,
.badge--unknown {
    background: rgba(148, 163, 184, 0.16);
    color: #475569;
}

.badge--mixed {
    background: rgba(245, 158, 11, 0.14);
    color: #92400e;
}

.intensity-pill {
    display: inline-flex;
    align-items: center;
    padding: 4px 8px;
    border-radius: 10px;
    background: rgba(15, 23, 42, 0.06);
    color: #0f172a;
    font-size: 0.8rem;
}

.cluster-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 14px;
}

.cluster-card {
    display: flex;
    flex-direction: column;
    gap: 12px;
    padding: 18px;
}

.cluster-desc {
    margin: 0;
    color: var(--text);
    font-size: 0.94rem;
    line-height: 1.7;
}

.chip-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.quote-box {
    margin: 0;
    padding: 12px 14px;
    border-left: 3px solid var(--primary);
    background: rgba(37, 99, 235, 0.06);
    border-radius: 12px;
    color: #0f172a;
    line-height: 1.7;
}

.empty-panel {
    padding: 18px;
}

.empty-panel__eyebrow {
    margin: 0 0 6px;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--primary);
}

.empty-panel__title {
    margin: 0;
    font-size: 1.02rem;
    font-weight: 800;
    color: var(--text-strong);
}

.empty-panel__desc {
    margin: 8px 0 0;
    color: var(--muted);
    font-size: 0.94rem;
    line-height: 1.7;
}

.empty-panel__note {
    margin-top: 12px;
    padding: 12px 14px;
    border-radius: 12px;
    background: rgba(37, 99, 235, 0.06);
    border: 1px solid rgba(37, 99, 235, 0.10);
    color: #1d4ed8;
    font-size: 0.88rem;
    line-height: 1.6;
}

.empty-inline {
    padding: 14px 16px;
    color: var(--muted);
    font-size: 0.9rem;
    line-height: 1.6;
    background: rgba(255, 255, 255, 0.9);
}

.footer-note {
    margin-top: 18px;
    padding-top: 16px;
    border-top: 1px solid rgba(148, 163, 184, 0.22);
    color: var(--muted);
    font-size: 0.88rem;
    line-height: 1.7;
}

@media (max-width: 1100px) {
    .hero-card {
        flex-direction: column;
    }

    .hero-meta {
        min-width: 0;
    }

    .summary-grid,
    .cluster-grid {
        grid-template-columns: 1fr 1fr;
    }

    .data-row,
    .detail-row {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 720px) {
    .app-shell {
        padding: 16px 12px 28px;
    }

    .hero-card,
    .panel-card {
        padding: 18px;
        border-radius: 22px;
    }

    .hero-meta,
    .summary-grid,
    .cluster-grid {
        grid-template-columns: 1fr;
    }

    .summary-head,
    .report-section-head {
        flex-direction: column;
        align-items: flex-start;
    }

    .action-row {
        gap: 8px;
    }
}

/* Editorial overrides */
:root {
    --surface: rgba(255, 255, 255, 0.92);
    --surface-strong: rgba(255, 255, 255, 0.98);
    --border: rgba(15, 23, 42, 0.09);
    --border-strong: rgba(29, 78, 216, 0.18);
    --primary: #1d4ed8;
    --primary-soft: rgba(29, 78, 216, 0.12);
    --primary-soft-2: rgba(29, 78, 216, 0.06);
    --shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
}

body {
    font-family: "Aptos", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
    color: var(--text);
}

.gradio-container {
    background:
        radial-gradient(circle at top left, rgba(29, 78, 216, 0.14), transparent 26%),
        radial-gradient(circle at top right, rgba(15, 23, 42, 0.08), transparent 22%),
        linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
    color: var(--text);
}

.app-shell {
    width: min(100%, 1440px);
    margin: 0 auto;
    padding: 24px 18px 40px;
    gap: 18px;
}

.hero-card {
    position: relative;
    display: grid;
    grid-template-columns: minmax(0, 1.55fr) minmax(260px, 0.85fr);
    gap: 24px;
    padding: 28px 30px 30px 34px;
    border-radius: 30px;
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(248, 250, 252, 0.96));
    color: var(--text-strong);
    box-shadow: 0 24px 60px rgba(15, 23, 42, 0.10);
    border: 1px solid rgba(15, 23, 42, 0.10);
    overflow: hidden;
}

.hero-card::before {
    content: "";
    position: absolute;
    inset: 0 auto 0 0;
    width: 8px;
    background: linear-gradient(180deg, #1d4ed8, #0f172a);
}

.hero-card::after {
    content: "";
    position: absolute;
    inset: auto -64px -72px auto;
    width: 240px;
    height: 240px;
    border-radius: 999px;
    background: radial-gradient(circle, rgba(29, 78, 216, 0.12), transparent 68%);
    pointer-events: none;
}

.hero-eyebrow {
    color: var(--primary);
}

.hero-title {
    font-family: "Georgia", "Songti SC", "STSong", serif;
    font-size: clamp(2.1rem, 4vw, 3.4rem);
    line-height: 1.05;
    font-weight: 400;
    color: var(--text-strong);
    letter-spacing: 0.01em;
}

.hero-lead {
    color: var(--text);
    line-height: 1.82;
}

.hero-meta {
    display: flex;
    flex-direction: column;
    gap: 10px;
    align-self: center;
    min-width: 0;
}

.hero-note {
    padding: 14px 16px;
    border-radius: 18px;
    background: rgba(15, 23, 42, 0.03);
    border: 1px solid rgba(15, 23, 42, 0.08);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.72);
}

.hero-note__label {
    font-size: 0.94rem;
    font-weight: 800;
    color: var(--text-strong);
}

.hero-note__text {
    margin-top: 4px;
    font-size: 0.86rem;
    line-height: 1.6;
    color: var(--muted);
}

.hero-note code,
.empty-panel__note code,
.section-desc code {
    display: inline-flex;
    align-items: center;
    padding: 0.14em 0.4em;
    border-radius: 6px;
    background: rgba(29, 78, 216, 0.08);
    border: 1px solid rgba(29, 78, 216, 0.12);
    color: var(--primary);
    font-family: inherit;
    font-size: 0.9em;
}

.workspace-grid {
    display: grid;
    grid-template-columns: minmax(0, 0.94fr) minmax(0, 1.06fr);
    gap: 18px;
    align-items: flex-start;
}

.panel-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-xl);
    box-shadow: var(--shadow);
    padding: 20px;
    overflow: hidden;
}

.panel-card .html-container {
    padding: 0 !important;
    background: transparent !important;
}

.panel-card .html-container.pending,
.detail-accordion .html-container.pending {
    opacity: 1 !important;
}

.panel-card .html-container .prose {
    margin: 0 !important;
    padding: 0 !important;
    background: transparent !important;
}

.panel-card .html-container .prose > :first-child {
    margin-top: 0 !important;
}

.panel-card .styler,
.panel-card .form {
    background: transparent !important;
}

.section-header {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin: 0 0 16px;
}

.section-title {
    margin: 0;
    font-size: 1.05rem;
    font-weight: 800;
    color: var(--text-strong);
    line-height: 1.3;
}

.section-desc {
    margin: 0;
    color: var(--muted);
    font-size: 0.94rem;
    line-height: 1.72;
}

.action-row {
    margin-top: 10px;
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}

.action-row > * {
    flex: 1 1 calc(50% - 5px);
    min-width: 140px;
}

.action-row .gr-button {
    min-height: 46px !important;
    border-radius: 14px !important;
}

.action-row .gr-button-primary {
    flex-basis: 100%;
    box-shadow: 0 10px 24px rgba(29, 78, 216, 0.18);
}

.detail-nav {
    display: flex;
    flex-direction: column;
    gap: 12px;
    margin-bottom: 18px;
    padding: 16px 16px 14px;
    border-radius: 20px;
    background: linear-gradient(180deg, rgba(248, 250, 252, 0.96), rgba(255, 255, 255, 0.98));
    border: 1px solid rgba(148, 163, 184, 0.16);
}

.detail-nav__head {
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.detail-nav__eyebrow {
    font-size: 0.74rem;
    font-weight: 800;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--primary);
}

.detail-nav__title {
    font-size: 0.98rem;
    font-weight: 800;
    color: var(--text-strong);
}

.detail-nav__chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.detail-nav__chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    min-height: 36px;
    padding: 0 12px;
    border-radius: 999px;
    text-decoration: none;
    border: 1px solid rgba(37, 99, 235, 0.14);
    background: rgba(37, 99, 235, 0.06);
    color: #1737b5;
    font-size: 0.84rem;
    font-weight: 700;
    transition: transform 140ms ease, background 140ms ease, border-color 140ms ease;
}

.detail-nav__chip:hover,
.detail-nav__chip:focus-visible {
    transform: translateY(-1px);
    background: rgba(37, 99, 235, 0.10);
    border-color: rgba(37, 99, 235, 0.22);
}

.detail-nav__chip-index {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 1.5em;
    color: var(--primary);
}

.detail-nav__hint {
    font-size: 0.9rem;
    line-height: 1.7;
    color: var(--muted);
}

.detail-stack {
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.detail-accordion {
    scroll-margin-top: 16px;
}

.detail-accordion details,
.detail-accordion {
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.92);
    overflow: hidden;
}

.detail-accordion summary {
    list-style: none;
    cursor: pointer;
    padding: 14px 16px;
    font-weight: 800;
    color: var(--text-strong);
    background: linear-gradient(180deg, rgba(248, 250, 252, 0.98), rgba(255, 255, 255, 0.96));
    border-bottom: 1px solid transparent;
}

.detail-accordion summary::-webkit-details-marker {
    display: none;
}

.detail-accordion details[open] summary,
.detail-accordion[open] summary {
    border-bottom-color: rgba(148, 163, 184, 0.14);
}

.detail-accordion summary:hover {
    background: linear-gradient(180deg, rgba(241, 245, 249, 0.96), rgba(255, 255, 255, 0.98));
}

.detail-accordion summary:focus-visible {
    outline: 3px solid rgba(29, 78, 216, 0.22);
    outline-offset: -3px;
}

.detail-accordion [role="region"],
.detail-accordion details > *:not(summary) {
    padding: 16px;
}

.detail-accordion .empty-panel,
.detail-accordion .report-stack {
    margin: 0;
}

button:focus-visible,
input:focus-visible,
textarea:focus-visible,
select:focus-visible,
[role="tab"]:focus-visible {
    outline: 3px solid rgba(29, 78, 216, 0.32);
    outline-offset: 2px;
}

.summary-panel {
    display: flex;
    flex-direction: column;
    gap: 14px;
}

.summary-copy {
    color: var(--muted);
    font-size: 0.92rem;
    line-height: 1.72;
}

.summary-ledger {
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.summary-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 16px;
    align-items: center;
    padding: 13px 16px;
    border-radius: 18px;
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(247, 248, 250, 0.96));
    border: 1px solid rgba(15, 23, 42, 0.08);
}

.summary-row__meta {
    min-width: 0;
}

.summary-row__label {
    font-size: 0.9rem;
    font-weight: 800;
    color: var(--text-strong);
    line-height: 1.3;
}

.summary-row__hint {
    margin-top: 4px;
    font-size: 0.84rem;
    line-height: 1.5;
    color: var(--muted);
}

.summary-row__value {
    font-size: 1.3rem;
    font-weight: 800;
    color: var(--primary);
    white-space: nowrap;
}

.workflow-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.workflow-chip,
.chip {
    display: inline-flex;
    align-items: center;
    min-height: 32px;
    padding: 0 11px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 700;
    line-height: 1;
    border: 1px solid transparent;
}

.workflow-chip--pending {
    background: rgba(148, 163, 184, 0.12);
    color: var(--muted);
    border-color: rgba(148, 163, 184, 0.14);
}

.workflow-chip--active {
    background: rgba(29, 78, 216, 0.10);
    color: #1737b5;
    border-color: rgba(29, 78, 216, 0.20);
}

.workflow-chip--done {
    background: rgba(15, 23, 42, 0.05);
    color: var(--text-strong);
    border-color: rgba(15, 23, 42, 0.08);
}

.chip {
    background: rgba(29, 78, 216, 0.08);
    color: #1737b5;
    border-color: rgba(29, 78, 216, 0.08);
}

.chip--empty {
    background: rgba(148, 163, 184, 0.16);
    color: #475569;
    border-color: rgba(148, 163, 184, 0.10);
}

.module-divider {
    height: 1px;
    margin: 4px 0 2px;
    background: linear-gradient(90deg, transparent, rgba(148, 163, 184, 0.34), transparent);
}

.footer-note {
    margin-top: 18px;
    padding-top: 16px;
    border-top: 1px solid rgba(148, 163, 184, 0.22);
    color: var(--muted);
    font-size: 0.88rem;
    line-height: 1.7;
}

.insights-markdown {
    background: transparent !important;
}

.insights-markdown .prose,
.insights-markdown .markdown {
    max-width: none !important;
    padding: 0 !important;
    background: transparent !important;
}

.insights-markdown .prose p:last-child {
    margin-bottom: 0;
}

footer {
    display: none !important;
}

@media (max-width: 1100px) {
    .hero-card {
        grid-template-columns: 1fr;
    }

    .workspace-grid {
        grid-template-columns: 1fr;
    }

    .hero-meta {
        align-self: stretch;
    }

    .summary-row,
    .data-row,
    .detail-row {
        grid-template-columns: 1fr;
    }

    .summary-row__value {
        justify-self: start;
    }

    .cluster-grid {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 720px) {
    .app-shell {
        padding: 16px 12px 28px;
    }

    .hero-card,
    .panel-card {
        padding: 18px;
        border-radius: 22px;
    }

    .hero-title {
        font-size: clamp(1.55rem, 8.5vw, 2.25rem);
        line-height: 1.08;
    }

    .hero-meta {
        gap: 8px;
    }

    .action-row > * {
        flex-basis: 100%;
    }

    .summary-head,
    .report-section-head {
        flex-direction: column;
        align-items: flex-start;
    }

    .summary-badge,
    .report-count {
        width: fit-content;
    }

}
"""

def build_default_outputs():
    return (
        build_summary_placeholder(),
        build_empty_panel("主题编码", "当前还没有主题编码结果。", "点击开始分析后，这里会按问题展示主题标签、原文引述和研究员备注。"),
        build_empty_panel("情感分析", "当前还没有情感分析结果。", "点击开始分析后，这里会显示每段访谈的整体情感和细分方面。"),
        build_empty_panel("亲和图聚类", "当前还没有聚类结果。", "点击开始分析后，这里会展示按主题归纳后的分组卡片。"),
        "*等待分析...*",
    )


def load_demo_data():
    return (DEMO_TRANSCRIPT,) + build_default_outputs()


def clear_all_outputs():
    return ("",) + build_default_outputs()


with gr.Blocks(title="用户访谈分析工作台") as app:
    with gr.Column(elem_classes="app-shell"):
        gr.HTML(
            """
            <section class="hero-card">
              <div class="hero-copy">
                <div class="hero-eyebrow">用户研究分析工作台</div>
                <h1 class="hero-title">用户访谈自动化分析</h1>
                <p class="hero-lead">
                  面向访谈转录文本的结构化分析界面。结果按阶段逐步回填，研究员随时可以停在任一节点核对原文证据。
                </p>
              </div>
              <div class="hero-meta">
                <div class="hero-note">
                  <div class="hero-note__label">后端托管密钥</div>
                  <div class="hero-note__text"><code>DASHSCOPE_API_KEY</code> 直接写在脚本顶部，不进前端面板。</div>
                </div>
                <div class="hero-note">
                  <div class="hero-note__label">分段结果回填</div>
                  <div class="hero-note__text">主题编码、情感、聚类和洞察会按阶段依次刷新。</div>
                </div>
                <div class="hero-note">
                  <div class="hero-note__label">原文证据优先</div>
                  <div class="hero-note__text">所有结论都保留引述，方便你继续做人工判断。</div>
                </div>
              </div>
            </section>
            """
        )

        with gr.Row(elem_classes="workspace-grid"):
            # ===== 左侧：输入区 =====
            with gr.Column(scale=5):
                with gr.Group(elem_classes="panel-card"):
                    gr.HTML(
                        render_section_header(
                            "分析配置",
                            "模型在前端可选，密钥由脚本顶部的 DASHSCOPE_API_KEY 常量提供，前端不暴露。",
                        )
                    )
                    model_choice = gr.Dropdown(
                        choices=["qwen-plus", "qwen-turbo", "qwen-max"],
                        value="qwen-plus",
                        label="模型选择",
                        info="qwen-turbo 更快 | qwen-plus 均衡 | qwen-max 更强",
                    )

                with gr.Group(elem_classes="panel-card"):
                    gr.HTML(
                        render_section_header(
                            "访谈文本",
                            "支持标准 Q&A 格式，也支持自由段落格式。建议保留受访者原话，方便后续追溯证据。",
                        )
                    )
                    transcript_input = gr.Textbox(
                        label="访谈文本",
                        placeholder="粘贴访谈转录文本...\n\n支持格式：\nQ1：问题\nP01：回答\n\n也支持自由段落格式",
                        lines=18,
                        max_lines=50,
                    )
                    with gr.Row(elem_classes="action-row"):
                        demo_btn = gr.Button("加载示例", variant="secondary")
                        clear_btn = gr.Button("清空内容", variant="secondary")
                        analyze_btn = gr.Button("开始分析", variant="primary", scale=2)

            # ===== 右侧：结果区 =====
            with gr.Column(scale=7):
                with gr.Group(elem_classes="panel-card"):
                    gr.HTML(
                        render_section_header(
                            "结果总览",
                            "先看总览，再向下查看编码、情感、聚类和洞察。",
                        )
                    )
                    summary_output = gr.HTML(value=build_summary_placeholder())

                with gr.Group(elem_classes="panel-card"):
                    gr.HTML(
                        render_section_header(
                            "结果明细",
                            "通过目录快速跳转，并按模块折叠查看，适合更长的访谈。",
                        )
                    )
                    gr.HTML(build_detail_directory())
                    with gr.Column(elem_classes="detail-stack"):
                        with gr.Accordion("01 主题编码", open=True, elem_id="detail-codes", elem_classes="detail-accordion"):
                            codes_output = gr.HTML(value=build_empty_panel("主题编码", "当前还没有主题编码结果。", "点击开始分析后，这里会按问题展示主题标签、原文引述和研究员备注。"))
                        with gr.Accordion("02 情感分析", open=False, elem_id="detail-sentiment", elem_classes="detail-accordion"):
                            sentiment_output = gr.HTML(value=build_empty_panel("情感分析", "当前还没有情感分析结果。", "点击开始分析后，这里会显示每段访谈的整体情感和细分方面。"))
                        with gr.Accordion("03 亲和图聚类", open=False, elem_id="detail-affinity", elem_classes="detail-accordion"):
                            affinity_output = gr.HTML(value=build_empty_panel("亲和图聚类", "当前还没有聚类结果。", "点击开始分析后，这里会展示按主题归纳后的分组卡片。"))
                        with gr.Accordion("04 关键洞察", open=False, elem_id="detail-insights", elem_classes="detail-accordion"):
                            insights_output = gr.Markdown(value="*等待分析后生成研究洞察。*", elem_classes="insights-markdown")

        gr.HTML(
            """
            <div class="footer-note">
              设计原则：人机协作优先，所有结论保留原文证据，界面围绕研究流程而不是功能堆叠展开。
            </div>
            """
        )

    # ===== 事件绑定 =====
    demo_btn.click(
        fn=load_demo_data,
        outputs=[transcript_input, summary_output, codes_output, sentiment_output, affinity_output, insights_output],
    )
    clear_btn.click(
        fn=clear_all_outputs,
        outputs=[transcript_input, summary_output, codes_output, sentiment_output, affinity_output, insights_output],
    )
    analyze_btn.click(
        fn=run_analysis,
        inputs=[transcript_input, model_choice],
        outputs=[summary_output, codes_output, sentiment_output, affinity_output, insights_output],
    )


if __name__ == "__main__":
    app.queue(default_concurrency_limit=1)
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(primary_hue="blue"),
        css=CUSTOM_CSS
    )
