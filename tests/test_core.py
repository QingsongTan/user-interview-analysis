import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ===== Task 1: API Key 安全 =====

def test_api_key_not_hardcoded():
    with open("interview_analysis_app.py", encoding="utf-8") as f:
        source = f.read()
    assert "sk-9721777521424690" not in source, "API Key 仍硬编码在源文件中"


# ===== Task 2: 健壮 JSON 解析 =====

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


# ===== Task 3: 研究问题输入 =====

def test_build_research_context_empty():
    from interview_analysis_app import build_research_context
    result = build_research_context("")
    assert result == ""


def test_build_research_context_with_question():
    from interview_analysis_app import build_research_context
    result = build_research_context("用户为什么流失？")
    assert "用户为什么流失" in result
    assert "研究目标" in result


# ===== Task 4: 报告导出 =====

def test_build_export_report_contains_sections():
    from interview_analysis_app import build_export_report
    codes = {"Q1": [{"theme": "价格", "quote": "便宜", "note": "促销驱动"}]}
    sentiments = [{"id": "Q1", "overall": "正面", "details": []}]
    affinity = {
        "clusters": [{
            "group_name": "价格敏感",
            "description": "desc",
            "themes": ["价格"],
            "representative_quote": "便宜"
        }]
    }
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


# ===== Task 5: 多访谈对比 =====

def test_synthesize_multi_interview_basic():
    from interview_analysis_app import synthesize_multi_interview_data
    per_interview = [
        {"label": "受访者A", "themes": ["价格敏感", "通知不稳定"], "sentiment_summary": "混合", "segments_count": 3},
        {"label": "受访者B", "themes": ["价格敏感", "退货麻烦"], "sentiment_summary": "负面", "segments_count": 4},
    ]
    result = synthesize_multi_interview_data(per_interview)
    assert "价格敏感" in result
    assert "受访者A" in result


def test_collect_interview_summary_empty_skipped():
    from interview_analysis_app import collect_interview_summary
    summary = collect_interview_summary("", "受访者A")
    assert summary is None
