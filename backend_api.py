import os
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    transcript: str = Field(..., min_length=1)
    model: str = "qwen-plus"
    research_question: str = ""


class ExportReportRequest(BaseModel):
    all_codes: dict[str, list[dict[str, Any]]]
    sentiments: list[dict[str, Any]]
    affinity: dict[str, Any]
    insights: str = ""
    research_question: str = ""


app = FastAPI(title="访谈分析 API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def load_analysis_module():
    import interview_analysis_app as analysis

    return analysis


def build_analysis_payload(transcript: str, model: str, research_question: str) -> dict[str, Any]:
    if not transcript or not transcript.strip():
        raise HTTPException(status_code=400, detail="transcript 不能为空")

    analysis = load_analysis_module()

    analysis.MODEL = model

    try:
        segments = analysis.preprocess_transcript(transcript)
        if not segments:
            raise HTTPException(status_code=400, detail="未能从文本中提取到有效的问答对，请检查格式")

        all_codes = analysis.code_themes(segments, research_question=research_question)
        sentiments = analysis.analyze_sentiment(segments, research_question=research_question)
        affinity = analysis.cluster_themes(all_codes)
        insights = analysis.generate_insights(affinity, sentiments, research_question=research_question)
        report_markdown = analysis.build_export_report(
            all_codes=all_codes,
            sentiments=sentiments,
            affinity=affinity,
            insights=insights,
            research_question=research_question,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"分析失败：{exc}") from exc

    return {
        "model": model,
        "research_question": research_question,
        "segments_count": len(segments),
        "total_themes": sum(len(items) for items in all_codes.values()),
        "total_clusters": len(affinity.get("clusters", [])),
        "sentiments_count": len(sentiments),
        "all_codes": all_codes,
        "sentiments": sentiments,
        "affinity": affinity,
        "insights": insights,
        "report_markdown": report_markdown,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "api_key_configured": bool(os.getenv("DASHSCOPE_API_KEY", "").strip()),
        "default_model": "qwen-plus",
    }


@app.post("/api/analyze")
def analyze_interview(request: AnalyzeRequest) -> dict[str, Any]:
    return build_analysis_payload(
        transcript=request.transcript,
        model=request.model,
        research_question=request.research_question,
    )


@app.post("/api/export-report")
def export_report(request: ExportReportRequest) -> dict[str, str]:
    analysis = load_analysis_module()

    try:
        markdown = analysis.build_export_report(
            all_codes=request.all_codes,
            sentiments=request.sentiments,
            affinity=request.affinity,
            insights=request.insights,
            research_question=request.research_question,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"导出失败：{exc}") from exc

    return {"markdown": markdown}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend_api:app", host="127.0.0.1", port=8000, reload=True)
