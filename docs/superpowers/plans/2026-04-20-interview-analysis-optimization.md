# Interview Analysis App 优化实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在单文件架构内完成七项优化，使工具从单人访谈演示品升级为可演示真实用研工作流的原型。

**Architecture:** 保持 `interview_analysis_app.py` 单文件，不拆分模块。新增 `.env` 文件管理密钥，Gradio 界面增加"研究问题"输入和"多访谈对比"标签页，LLM 调用层加 tenacity 重试和 JSON 修复。

**Tech Stack:** Python 3.10+, Gradio 4.x, OpenAI SDK, tenacity, python-dotenv

---

## 设计约束与边界

| 约束 | 说明 |
|------|------|
| **单文件** | 不拆分模块，方便演示时直接讲解整个文件 |
| **不改底层流程** | 5 步分析管线（预处理→编码→情感→聚类→洞察）结构不变 |
| **不引入新 UI 框架** | 纯 Gradio，CSS 风格与现有保持一致 |
| **多访谈上限** | 最多同时分析 3 份访谈（演示场景，避免 API 超时） |
| **导出格式** | Markdown 纯文本，不依赖 WeasyPrint/docx 等重型库 |
| **向后兼容** | 原有单访谈分析流程行为不变 |

---

## 文件结构

| 文件 | 操作 | 说明 |
|------|------|------|
| `interview_analysis_app.py` | 修改 | 所有优化都在此文件 |
| `.env` | 新建 | 存放 DASHSCOPE_API_KEY |
| `.env.example` | 新建 | 模板文件，可提交到 git |
| `tests/test_core.py` | 新建 | 核心逻辑单元测试 |

---

## Task 1: API Key 安全（.env 文件）

**Files:**
- Modify: `interview_analysis_app.py:13-17`
- Create: `.env`
- Create: `.env.example`
- Create: `tests/test_core.py`

- [ ] **Step 1: 创建 .env 文件**

```
DASHSCOPE_API_KEY=sk-9721777521424690b1a7000db221b95b
```

- [ ] **Step 2: 创建 .env.example**

```
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

- [ ] **Step 3: 写失败测试**

在 `tests/test_core.py` 中写：

```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_api_key_not_hardcoded():
    with open("interview_analysis_app.py", encoding="utf-8") as f:
        source = f.read()
    assert "sk-9721777521424690" not in source, "API Key 仍硬编码在源文件中"
```

- [ ] **Step 4: 运行测试确认失败**

```bash
cd "c:/Users/17675/Desktop/实习/腾讯用研"
python -m pytest tests/test_core.py::test_api_key_not_hardcoded -v
```

期望：FAIL（因为 key 现在还在文件里）

- [ ] **Step 5: 修改 interview_analysis_app.py 顶部**

将：
```python
from openai import OpenAI

# ========== LLM 配置 ==========
client = None
client_api_key = None
MODEL = "qwen-plus"
DASHSCOPE_API_KEY = "sk-9721777521424690b1a7000db221b95b"
```

替换为：
```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ========== LLM 配置 ==========
client = None
client_api_key = None
MODEL = "qwen-plus"
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
```

- [ ] **Step 6: 运行测试确认通过**

```bash
python -m pytest tests/test_core.py::test_api_key_not_hardcoded -v
```

期望：PASS

- [ ] **Step 7: Commit**

```bash
git add .env.example tests/test_core.py interview_analysis_app.py
git commit -m "feat: 将API Key迁移至.env文件，避免密钥硬编码"
```

---

## Task 2: LLM 重试机制 + 健壮 JSON 解析

**Files:**
- Modify: `interview_analysis_app.py`（`call_llm` 函数 + JSON 解析处）
- Test: `tests/test_core.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_core.py` 追加：

```python
import json

def test_safe_parse_json_array_valid():
    from interview_analysis_app import safe_parse_json_array
    result = safe_parse_json_array('[{"theme": "价格敏感", "quote": "便宜30%", "note": "促销驱动"}]')
    assert len(result) == 1
    assert result[0]["theme"] == "价格敏感"

def test_safe_parse_json_array_with_markdown_fence():
    from interview_analysis_app import safe_parse_json_array
    raw = '```json\n[{"theme": "易用性", "quote": "很方便", "note": ""}]\n```'
    result = safe_parse_json_array(raw)
    assert len(result) == 1

def test_safe_parse_json_array_invalid_returns_empty():
    from interview_analysis_app import safe_parse_json_array
    result = safe_parse_json_array("这不是JSON")
    assert result == []

def test_safe_parse_json_object_valid():
    from interview_analysis_app import safe_parse_json_object
    raw = '{"clusters": [{"group_name": "体验痛点"}]}'
    result = safe_parse_json_object(raw)
    assert "clusters" in result

def test_safe_parse_json_object_invalid_returns_empty_dict():
    from interview_analysis_app import safe_parse_json_object
    result = safe_parse_json_object("not json at all")
    assert result == {}
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest tests/test_core.py -k "safe_parse" -v
```

期望：全部 FAIL（函数尚未定义）

- [ ] **Step 3: 在 interview_analysis_app.py 中替换旧的解析常量和 call_llm**

找到文件中 `_QA_PATTERN` 所在区域（约第 219 行），在 `_QA_PATTERN` 定义**之前**插入以下代码：

```python
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ========== 健壮 JSON 解析 ==========

_JSON_FENCE_PATTERN = re.compile(r'```(?:json)?\s*([\s\S]*?)```', re.DOTALL)


def safe_parse_json_array(text: str) -> list:
    """从 LLM 输出中安全提取 JSON 数组，处理 markdown 围栏和格式噪声。"""
    # 优先尝试去掉 markdown 围栏
    fence_match = _JSON_FENCE_PATTERN.search(text)
    candidates = [fence_match.group(1).strip()] if fence_match else []
    # 再尝试 regex 提取裸数组
    raw_match = re.search(r'\[.*\]', text, re.DOTALL)
    if raw_match:
        candidates.append(raw_match.group())
    for candidate in candidates:
        try:
            result = json.loads(candidate)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    return []


def safe_parse_json_object(text: str) -> dict:
    """从 LLM 输出中安全提取 JSON 对象，处理 markdown 围栏和格式噪声。"""
    fence_match = _JSON_FENCE_PATTERN.search(text)
    candidates = [fence_match.group(1).strip()] if fence_match else []
    raw_match = re.search(r'\{.*\}', text, re.DOTALL)
    if raw_match:
        candidates.append(raw_match.group())
    for candidate in candidates:
        try:
            result = json.loads(candidate)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
    return {}
```

- [ ] **Step 4: 替换 call_llm 加入 tenacity 重试**

将原有的：
```python
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
```

替换为：
```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
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
```

- [ ] **Step 5: 替换旧的 _JSON_ARRAY_PATTERN / _JSON_OBJECT_PATTERN 使用点**

将 `code_themes` 中的：
```python
json_match = _JSON_ARRAY_PATTERN.search(result)
return seg['id'], json.loads(json_match.group()) if json_match else []
```
替换为：
```python
return seg['id'], safe_parse_json_array(result)
```

将 `analyze_sentiment` 中的：
```python
json_match = _JSON_ARRAY_PATTERN.search(result)
return json.loads(json_match.group()) if json_match else []
```
替换为：
```python
return safe_parse_json_array(result)
```

将 `cluster_themes` 中的：
```python
json_match = _JSON_OBJECT_PATTERN.search(result)
return json.loads(json_match.group()) if json_match else {}
```
替换为：
```python
return safe_parse_json_object(result)
```

删除不再使用的两个 pattern 常量：
```python
_JSON_ARRAY_PATTERN = re.compile(r'\[.*\]', re.DOTALL)
_JSON_OBJECT_PATTERN = re.compile(r'\{.*\}', re.DOTALL)
```

- [ ] **Step 6: 运行测试确认通过**

```bash
python -m pytest tests/test_core.py -k "safe_parse" -v
```

期望：全部 PASS

- [ ] **Step 7: Commit**

```bash
git add interview_analysis_app.py tests/test_core.py
git commit -m "feat: 添加LLM重试机制和健壮JSON解析，消除静默失败"
```

---

## Task 3: 研究问题输入框

**Files:**
- Modify: `interview_analysis_app.py`（`code_themes`, `analyze_sentiment`, `generate_insights`, `run_analysis` 函数 + UI 区域）
- Test: `tests/test_core.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_core.py` 追加：

```python
def test_build_research_context_empty():
    from interview_analysis_app import build_research_context
    result = build_research_context("")
    assert result == ""

def test_build_research_context_with_question():
    from interview_analysis_app import build_research_context
    result = build_research_context("用户为什么流失？")
    assert "用户为什么流失" in result
    assert "研究目标" in result
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest tests/test_core.py -k "research_context" -v
```

期望：FAIL

- [ ] **Step 3: 在 interview_analysis_app.py 的 # ========== 分析模块 ========== 之前添加辅助函数**

```python
def build_research_context(research_question: str) -> str:
    """将研究目标格式化为 prompt 前缀，空时返回空字符串。"""
    if not research_question or not research_question.strip():
        return ""
    return f"\n\n【本次研究目标】\n{research_question.strip()}\n请在分析时优先关注与上述目标相关的信息。\n"
```

- [ ] **Step 4: 修改 code_themes 函数签名和 system prompt**

将：
```python
def code_themes(segments: list[dict], progress=None) -> dict:
    system = """你是一位专业的用户研究员，擅长对用户访谈文本进行主题编码。
请严格按照以下规则进行编码：
1. 每段回答可能包含多个主题，请逐一提取
2. 每个主题必须配对原文引述（直接引用受访者原话）
3. 主题标签要简洁、一致，便于后续聚类
4. 输出严格的 JSON 格式"""
```

替换为：
```python
def code_themes(segments: list[dict], research_question: str = "", progress=None) -> dict:
    research_ctx = build_research_context(research_question)
    system = f"""你是一位专业的用户研究员，擅长对用户访谈文本进行主题编码。{research_ctx}
请严格按照以下规则进行编码：
1. 每段回答可能包含多个主题，请逐一提取
2. 每个主题必须配对原文引述（直接引用受访者原话）
3. 主题标签要简洁、一致，便于后续聚类
4. 输出严格的 JSON 格式"""
```

- [ ] **Step 5: 修改 analyze_sentiment 函数签名**

将：
```python
def analyze_sentiment(segments: list[dict]) -> list[dict]:
    system = """你是用户研究情感分析专家。
请对每段用户回答进行情感标注。
注意区分"对产品功能的情感"和"对体验问题的情感"，同一段回答中可能包含多种情感。"""
```

替换为：
```python
def analyze_sentiment(segments: list[dict], research_question: str = "") -> list[dict]:
    research_ctx = build_research_context(research_question)
    system = f"""你是用户研究情感分析专家。{research_ctx}
请对每段用户回答进行情感标注。
注意区分"对产品功能的情感"和"对体验问题的情感"，同一段回答中可能包含多种情感。"""
```

- [ ] **Step 6: 修改 generate_insights 函数签名**

将：
```python
def generate_insights(affinity: dict, sentiments: list) -> str:
```

替换为：
```python
def generate_insights(affinity: dict, sentiments: list, research_question: str = "") -> str:
    research_ctx = build_research_context(research_question)
```

在该函数的 `user_prompt` 变量定义**之前**，将原来的 `user_prompt = f"""基于以下...` 修改，在开头加入研究目标：

```python
    user_prompt = f"""基于以下访谈分析结果，请生成研究洞察报告。{research_ctx}
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
```

- [ ] **Step 7: 修改 run_analysis 函数签名和内部调用**

将：
```python
def run_analysis(transcript: str, model_choice: str, progress=gr.Progress()):
```

替换为：
```python
def run_analysis(transcript: str, model_choice: str, research_question: str = "", progress=gr.Progress()):
```

将函数内：
```python
    all_codes = code_themes(segments, progress=progress)
```
替换为：
```python
    all_codes = code_themes(segments, research_question=research_question, progress=progress)
```

将函数内：
```python
        future_sentiment = executor.submit(analyze_sentiment, segments)
```
替换为：
```python
        future_sentiment = executor.submit(analyze_sentiment, segments, research_question)
```

将函数内：
```python
    insights = generate_insights(affinity, sentiments)
```
替换为：
```python
    insights = generate_insights(affinity, sentiments, research_question)
```

- [ ] **Step 8: 在 Gradio UI 的左侧输入区添加研究问题输入框**

在 `transcript_input` 的 `gr.Textbox` 定义**之前**，在 `with gr.Group(elem_classes="panel-card"):` 内插入新的卡片区块，即在 `gr.HTML(render_section_header("访谈文本", ...))` **之前**插入：

```python
                with gr.Group(elem_classes="panel-card"):
                    gr.HTML(
                        render_section_header(
                            "研究目标（可选）",
                            "输入本次研究的核心问题或假设，AI 分析时会优先关注相关信息。留空则进行通用分析。",
                        )
                    )
                    research_question_input = gr.Textbox(
                        label="研究问题 / 研究假设",
                        placeholder="例：用户为何会在付款环节流失？/ 用户对新版消息通知功能的感知如何？",
                        lines=2,
                        max_lines=4,
                    )
```

- [ ] **Step 9: 更新事件绑定，将 research_question_input 加入 inputs**

将：
```python
    analyze_btn.click(
        fn=run_analysis,
        inputs=[transcript_input, model_choice],
        outputs=[summary_output, codes_output, sentiment_output, affinity_output, insights_output],
    )
```

替换为：
```python
    analyze_btn.click(
        fn=run_analysis,
        inputs=[transcript_input, model_choice, research_question_input],
        outputs=[summary_output, codes_output, sentiment_output, affinity_output, insights_output],
    )
```

- [ ] **Step 10: 运行测试确认通过**

```bash
python -m pytest tests/test_core.py -k "research_context" -v
```

期望：PASS

- [ ] **Step 11: Commit**

```bash
git add interview_analysis_app.py tests/test_core.py
git commit -m "feat: 添加研究目标输入框，引导AI聚焦特定研究问题"
```

---

## Task 4: 报告导出（Markdown 下载）

**Files:**
- Modify: `interview_analysis_app.py`（新增导出函数 + UI 按钮）
- Test: `tests/test_core.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_core.py` 追加：

```python
def test_build_export_report_contains_sections():
    from interview_analysis_app import build_export_report
    codes = {"Q1": [{"theme": "价格", "quote": "便宜", "note": "促销驱动"}]}
    sentiments = [{"id": "Q1", "overall": "正面", "details": []}]
    affinity = {"clusters": [{"group_name": "价格敏感", "description": "desc", "themes": ["价格"], "representative_quote": "便宜"}]}
    insights = "### 核心发现\n- 发现1"
    report = build_export_report(codes, sentiments, affinity, insights, research_question="为什么购买？")
    assert "## 研究目标" in report
    assert "## 主题编码" in report
    assert "## 情感分析" in report
    assert "## 亲和图聚类" in report
    assert "## 关键洞察" in report
    assert "价格" in report

def test_build_export_report_no_research_question():
    from interview_analysis_app import build_export_report
    report = build_export_report({}, [], {}, "", research_question="")
    assert "## 研究目标" not in report
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest tests/test_core.py -k "export_report" -v
```

期望：FAIL

- [ ] **Step 3: 在 # ========== 格式化输出 ========== 区域之后，主分析流程之前添加导出函数**

```python
def build_export_report(
    all_codes: dict,
    sentiments: list,
    affinity: dict,
    insights: str,
    research_question: str = "",
) -> str:
    from datetime import datetime
    lines = [f"# 用户访谈分析报告", f"", f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}", ""]
    if research_question and research_question.strip():
        lines += [f"## 研究目标", "", research_question.strip(), ""]
    lines += ["## 主题编码", ""]
    for qid, codes in all_codes.items():
        lines.append(f"### {qid}")
        for c in codes:
            lines.append(f"- **{c.get('theme', '')}**：「{c.get('quote', '')}」—— {c.get('note', '')}")
        lines.append("")
    lines += ["## 情感分析", ""]
    for s in sentiments:
        lines.append(f"### {s.get('id', '?')}（整体：{s.get('overall', '')}）")
        for d in s.get("details", []):
            lines.append(f"- {d.get('aspect', '')}｜{d.get('sentiment', '')}｜强度 {d.get('intensity', '')}｜「{d.get('evidence', '')}」")
        lines.append("")
    lines += ["## 亲和图聚类", ""]
    for idx, cluster in enumerate(affinity.get("clusters", []), 1):
        lines.append(f"### 分组 {idx:02d}：{cluster.get('group_name', '')}")
        lines.append(cluster.get("description", ""))
        lines.append(f"- 包含主题：{', '.join(cluster.get('themes', []))}")
        lines.append(f"- 代表引述：「{cluster.get('representative_quote', '')}」")
        lines.append("")
    lines += ["## 关键洞察", "", insights or "（尚未生成）", ""]
    return "\n".join(lines)
```

- [ ] **Step 4: 在 run_analysis 函数末尾，最后一个 yield 之前保存结果到全局变量**

在 `run_analysis` 函数顶部（`global MODEL` 所在行之后）添加：

```python
    global _last_report_data
    _last_report_data = {}
```

在 `run_analysis` 最后一行 `yield summary, codes_html, sentiment_html, affinity_html, insights` **之前**插入：

```python
    _last_report_data = {
        "all_codes": all_codes,
        "sentiments": sentiments,
        "affinity": affinity,
        "insights": insights,
        "research_question": research_question,
    }
```

在文件顶部（`client = None` 附近）添加：
```python
_last_report_data: dict = {}
```

- [ ] **Step 5: 添加导出回调函数**

在 `build_default_outputs` 函数**之前**添加：

```python
def export_report_file():
    import tempfile, os
    if not _last_report_data:
        raise gr.Error("请先完成一次分析，再导出报告。")
    content = build_export_report(
        all_codes=_last_report_data.get("all_codes", {}),
        sentiments=_last_report_data.get("sentiments", []),
        affinity=_last_report_data.get("affinity", {}),
        insights=_last_report_data.get("insights", ""),
        research_question=_last_report_data.get("research_question", ""),
    )
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", prefix="interview_report_",
        encoding="utf-8", delete=False
    )
    tmp.write(content)
    tmp.close()
    return tmp.name
```

- [ ] **Step 6: 在 UI 的 action-row 添加导出按钮和 File 组件**

在 `with gr.Row(elem_classes="action-row"):` 块中，`analyze_btn` **之后**添加：

```python
                        export_btn = gr.Button("导出报告", variant="secondary")
```

在 `with gr.Row(elem_classes="action-row"):` 块**之后**（同一个 `gr.Group` 内）添加：

```python
                    export_file = gr.File(label="下载报告", visible=False)
```

- [ ] **Step 7: 添加事件绑定（在 # ===== 事件绑定 ===== 区域）**

在现有绑定之后追加：

```python
    export_btn.click(
        fn=export_report_file,
        outputs=export_file,
    ).then(
        fn=lambda: gr.File(visible=True),
        outputs=export_file,
    )
```

- [ ] **Step 8: 运行测试确认通过**

```bash
python -m pytest tests/test_core.py -k "export_report" -v
```

期望：PASS

- [ ] **Step 9: Commit**

```bash
git add interview_analysis_app.py tests/test_core.py
git commit -m "feat: 添加报告导出功能，支持下载Markdown格式分析报告"
```

---

## Task 5: 多访谈批量对比分析

**Files:**
- Modify: `interview_analysis_app.py`（新增 `run_multi_analysis` 函数 + 新增 Gradio Tab）
- Test: `tests/test_core.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_core.py` 追加：

```python
def test_synthesize_multi_interview_basic():
    from interview_analysis_app import synthesize_multi_interview_data
    per_interview = [
        {"label": "受访者A", "themes": ["价格敏感", "通知不稳定"], "sentiment_summary": "混合"},
        {"label": "受访者B", "themes": ["价格敏感", "退货麻烦"], "sentiment_summary": "负面"},
    ]
    result = synthesize_multi_interview_data(per_interview)
    assert "价格敏感" in result
    assert "受访者A" in result

def test_collect_interview_summary_empty_skipped():
    from interview_analysis_app import collect_interview_summary
    summary = collect_interview_summary("", "受访者A")
    assert summary is None
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest tests/test_core.py -k "multi_interview or collect_interview" -v
```

期望：FAIL

- [ ] **Step 3: 添加辅助函数 collect_interview_summary 和 synthesize_multi_interview_data**

在 `build_export_report` 函数**之后**，`build_default_outputs` 之前添加：

```python
def collect_interview_summary(transcript: str, label: str) -> dict | None:
    """对单份访谈做轻量分析，返回可供跨访谈对比的摘要字典。"""
    if not transcript or not transcript.strip():
        return None
    segments = preprocess_transcript(transcript)
    if not segments:
        return None
    all_codes = code_themes(segments)
    themes = list({c["theme"] for codes in all_codes.values() for c in codes})
    sentiments = analyze_sentiment(segments)
    overall_counts = {}
    for s in sentiments:
        key = s.get("overall", "未知")
        overall_counts[key] = overall_counts.get(key, 0) + 1
    dominant = max(overall_counts, key=overall_counts.get) if overall_counts else "未知"
    return {"label": label, "themes": themes, "sentiment_summary": dominant, "segments_count": len(segments)}


def synthesize_multi_interview_data(per_interview: list[dict]) -> str:
    """将多份访谈的摘要列表格式化为供 LLM 综合分析的文本。"""
    lines = []
    for item in per_interview:
        lines.append(f"【{item['label']}】片段数：{item.get('segments_count', '?')}，整体情感：{item['sentiment_summary']}")
        lines.append(f"  主题标签：{', '.join(item['themes']) if item['themes'] else '无'}")
    return "\n".join(lines)


def run_multi_analysis(t1: str, t2: str, t3: str, research_question: str = "", progress=gr.Progress()):
    """对最多3份访谈并行分析，最后综合输出跨访谈洞察。"""
    inputs = [(t1, "受访者1"), (t2, "受访者2"), (t3, "受访者3")]
    active = [(t, label) for t, label in inputs if t and t.strip()]
    if not active:
        raise gr.Error("请至少输入一份访谈文本。")

    progress(0.05, desc=f"开始并行分析 {len(active)} 份访谈")

    with ThreadPoolExecutor(max_workers=len(active)) as executor:
        futures = {executor.submit(collect_interview_summary, t, label): label for t, label in active}
        results = []
        for future in as_completed(futures):
            summary = future.result()
            if summary:
                results.append(summary)

    if not results:
        raise gr.Error("所有访谈文本均未能解析，请检查格式。")

    progress(0.75, desc="综合跨访谈洞察")
    data_text = synthesize_multi_interview_data(results)
    research_ctx = build_research_context(research_question)
    system = "你是腾讯用户研究团队的高级研究员，正在对多份用户访谈进行横向对比分析。"
    user_prompt = f"""以下是对多位受访者的访谈分析摘要：{research_ctx}

{data_text}

请输出跨访谈综合洞察报告（Markdown）：

### 共性发现
（所有或多数受访者都提到的核心主题，频率≥2次的优先列出）

### 差异点
（不同受访者之间存在明显分歧的地方）

### 高优先级用户需求
（综合多方观点，按重要性排序）

### 建议后续跟进的问题
每条发现都要注明来自哪几位受访者。"""
    synthesis = call_llm(system, user_prompt)

    progress(1.0, desc="多访谈综合分析完成")

    summary_lines = [f"**共分析 {len(results)} 份访谈**\n"]
    for r in results:
        summary_lines.append(f"- {r['label']}：{r['segments_count']} 个片段，整体情感={r['sentiment_summary']}，主题 {len(r['themes'])} 个")
    summary_md = "\n".join(summary_lines)

    return summary_md, synthesis
```

- [ ] **Step 4: 在 Gradio app 中添加多访谈 Tab**

在 `with gr.Blocks(...) as app:` 的最外层 `with gr.Column(elem_classes="app-shell"):` 块内，将 hero card 和 workspace grid 包裹在 `gr.Tabs` 中：

将原有的：
```python
    with gr.Column(elem_classes="app-shell"):
        gr.HTML("""..hero-card...""")
        with gr.Row(elem_classes="workspace-grid"):
            ...
        gr.HTML("""..footer-note...""")
```

替换为（保留 hero HTML 不变，只在其下增加 Tabs）：

```python
    with gr.Column(elem_classes="app-shell"):
        gr.HTML("""...hero-card HTML 原样保留...""")

        with gr.Tabs():
            with gr.TabItem("单人访谈分析"):
                with gr.Row(elem_classes="workspace-grid"):
                    # ===== 左侧：输入区 ===== （原样保留所有左侧内容）
                    ...
                    # ===== 右侧：结果区 ===== （原样保留所有右侧内容）
                    ...

            with gr.TabItem("多访谈对比"):
                gr.HTML(render_section_header(
                    "多访谈横向对比",
                    "最多输入 3 份访谈，AI 会并行分析后综合对比，提炼共性与差异。",
                ))
                with gr.Row():
                    with gr.Column():
                        multi_rq_input = gr.Textbox(
                            label="研究目标（可选）",
                            placeholder="例：用户对新版功能的核心诉求是什么？",
                            lines=2,
                        )
                    with gr.Column():
                        multi_analyze_btn = gr.Button("开始多访谈对比分析", variant="primary")
                with gr.Row():
                    multi_t1 = gr.Textbox(label="访谈 1", placeholder="粘贴第一份访谈文本...", lines=12)
                    multi_t2 = gr.Textbox(label="访谈 2", placeholder="粘贴第二份访谈文本（可选）...", lines=12)
                    multi_t3 = gr.Textbox(label="访谈 3", placeholder="粘贴第三份访谈文本（可选）...", lines=12)
                with gr.Row():
                    multi_summary_output = gr.Markdown(label="访谈概览")
                    multi_synthesis_output = gr.Markdown(label="跨访谈综合洞察")

        gr.HTML("""..footer-note HTML 原样保留...""")
```

- [ ] **Step 5: 绑定多访谈分析按钮**

在事件绑定区域追加：

```python
    multi_analyze_btn.click(
        fn=run_multi_analysis,
        inputs=[multi_t1, multi_t2, multi_t3, multi_rq_input],
        outputs=[multi_summary_output, multi_synthesis_output],
    )
```

- [ ] **Step 6: 运行测试确认通过**

```bash
python -m pytest tests/test_core.py -k "multi_interview or collect_interview" -v
```

期望：PASS

- [ ] **Step 7: 运行全量测试**

```bash
python -m pytest tests/test_core.py -v
```

期望：全部 PASS

- [ ] **Step 8: 启动应用手动验证**

```bash
cd "c:/Users/17675/Desktop/实习/腾讯用研"
python interview_analysis_app.py
```

验证清单：
- [ ] 单人分析 Tab 功能不受影响（加载示例 → 分析）
- [ ] 研究问题输入后，洞察报告内容聚焦
- [ ] 导出按钮生成 .md 文件，内容包含所有模块
- [ ] 多访谈 Tab：仅填写访谈1，点击分析，能正常输出
- [ ] 多访谈 Tab：填写访谈1+2，能输出共性/差异分析

- [ ] **Step 9: Commit**

```bash
git add interview_analysis_app.py tests/test_core.py
git commit -m "feat: 添加多访谈横向对比功能，支持最多3份访谈并行分析与综合洞察"
```

---

## 计划自审清单

### Spec 覆盖检查

| 优化项 | 对应 Task | 状态 |
|--------|-----------|------|
| API Key 安全 | Task 1 | ✅ |
| LLM 重试机制 | Task 2 | ✅ |
| 健壮 JSON 解析 | Task 2 | ✅ |
| 研究问题输入 | Task 3 | ✅ |
| 报告导出 | Task 4 | ✅ |
| 多访谈对比 | Task 5 | ✅ |
| 人机协作标注 | ❌ 范围外 | 复杂度高，留待后续 |

### Placeholder 扫描
- 无 TBD / TODO
- 所有函数都有完整实现代码
- 所有测试都有具体 assert

### 类型一致性
- `safe_parse_json_array` → Task 2 定义，Task 2 中调用 ✅
- `safe_parse_json_object` → Task 2 定义，Task 2 中调用 ✅
- `build_research_context` → Task 3 定义，Task 3-5 中调用 ✅
- `build_export_report` → Task 4 定义，`export_report_file` 中调用 ✅
- `collect_interview_summary` → Task 5 定义，`run_multi_analysis` 中调用 ✅
- `synthesize_multi_interview_data` → Task 5 定义，`run_multi_analysis` 中调用 ✅
- `_last_report_data` 全局变量 → Task 4 定义，`run_analysis` 写入，`export_report_file` 读取 ✅
