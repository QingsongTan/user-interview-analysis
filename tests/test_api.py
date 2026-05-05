from types import SimpleNamespace

from fastapi.testclient import TestClient

import backend_api


client = TestClient(backend_api.app)


def make_stub_analysis_module():
    def preprocess_transcript(transcript: str):
        return [{"id": "Q1", "question": "你为什么购买？", "answer": transcript}]

    def code_themes(segments, research_question=""):
        assert research_question == "验证研究目标"
        return {
            "Q1": [
                {
                    "theme": "价格敏感",
                    "quote": segments[0]["answer"],
                    "note": "被价格驱动",
                }
            ]
        }

    def analyze_sentiment(segments, research_question=""):
        assert research_question == "验证研究目标"
        return [
            {
                "id": segments[0]["id"],
                "overall": "正面",
                "details": [
                    {
                        "aspect": "价格",
                        "sentiment": "正面",
                        "intensity": 4,
                        "evidence": segments[0]["answer"],
                    }
                ],
            }
        ]

    def cluster_themes(all_codes):
        return {
            "clusters": [
                {
                    "group_name": "价格价值",
                    "description": "用户认为产品价格合适",
                    "themes": [all_codes["Q1"][0]["theme"]],
                    "representative_quote": all_codes["Q1"][0]["quote"],
                }
            ]
        }

    def generate_insights(affinity, sentiments, research_question=""):
        assert research_question == "验证研究目标"
        return f"核心洞察：{affinity['clusters'][0]['group_name']}，整体情感 {sentiments[0]['overall']}"

    def build_export_report(all_codes, sentiments, affinity, insights, research_question=""):
        return "\n".join(
            [
                "# 用户访谈分析报告",
                f"研究目标：{research_question}",
                f"主题：{all_codes['Q1'][0]['theme']}",
                f"情感：{sentiments[0]['overall']}",
                f"聚类：{affinity['clusters'][0]['group_name']}",
                insights,
            ]
        )

    return SimpleNamespace(
        MODEL="qwen-plus",
        preprocess_transcript=preprocess_transcript,
        code_themes=code_themes,
        analyze_sentiment=analyze_sentiment,
        cluster_themes=cluster_themes,
        generate_insights=generate_insights,
        build_export_report=build_export_report,
    )


def test_health_check_reports_env_status(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "api_key_configured": True,
        "default_model": "qwen-plus",
    }


def test_analyze_interview_returns_structured_payload(monkeypatch):
    stub_module = make_stub_analysis_module()
    monkeypatch.setattr(backend_api, "load_analysis_module", lambda: stub_module)

    response = client.post(
        "/api/analyze",
        json={
            "transcript": "因为价格合适，而且下单很方便。",
            "model": "qwen-max",
            "research_question": "验证研究目标",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "qwen-max"
    assert data["segments_count"] == 1
    assert data["total_themes"] == 1
    assert data["total_clusters"] == 1
    assert data["sentiments_count"] == 1
    assert data["all_codes"]["Q1"][0]["theme"] == "价格敏感"
    assert "核心洞察" in data["insights"]
    assert "# 用户访谈分析报告" in data["report_markdown"]


def test_analyze_interview_rejects_empty_transcript(monkeypatch):
    monkeypatch.setattr(
        backend_api,
        "load_analysis_module",
        lambda: (_ for _ in ()).throw(AssertionError("空文本不应触发模块加载")),
    )

    response = client.post(
        "/api/analyze",
        json={"transcript": "   ", "model": "qwen-plus", "research_question": ""},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "transcript 不能为空"


def test_export_report_returns_markdown(monkeypatch):
    stub_module = make_stub_analysis_module()
    monkeypatch.setattr(backend_api, "load_analysis_module", lambda: stub_module)

    response = client.post(
        "/api/export-report",
        json={
            "all_codes": {"Q1": [{"theme": "价格敏感", "quote": "便宜", "note": "促销驱动"}]},
            "sentiments": [{"id": "Q1", "overall": "正面", "details": []}],
            "affinity": {
                "clusters": [
                    {
                        "group_name": "价格价值",
                        "description": "用户认为产品价格合适",
                        "themes": ["价格敏感"],
                        "representative_quote": "便宜",
                    }
                ]
            },
            "insights": "核心洞察：价格价值",
            "research_question": "验证研究目标",
        },
    )

    assert response.status_code == 200
    assert "# 用户访谈分析报告" in response.json()["markdown"]
    assert "研究目标：验证研究目标" in response.json()["markdown"]
