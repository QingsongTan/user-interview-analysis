import { Download, LoaderCircle, RefreshCcw, Sparkles } from "lucide-react";
import { useMemo, useState } from "react";

import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";

type CodeItem = {
  theme: string;
  quote: string;
  note: string;
};

type SentimentItem = {
  id: string;
  overall: string;
  details: Array<{
    aspect: string;
    sentiment: string;
    intensity: number;
    evidence: string;
  }>;
};

type AffinityResult = {
  clusters: Array<{
    group_name: string;
    description: string;
    themes: string[];
    representative_quote: string;
  }>;
};

type AnalyzeResponse = {
  model: string;
  segments_count: number;
  all_codes: Record<string, CodeItem[]>;
  sentiments: SentimentItem[];
  affinity: AffinityResult;
  insights: string;
  report_markdown: string;
};

const demoTranscript = `Q1：你平时每天大概花多长时间在抖音上，一般在什么场景下刷？
P01：通勤时刷得最多，晚上也会停不下来。

Q2：有没有让你觉得不爽或者体验不好的时候？
P01：广告和重复推荐最打断体验。`;

const emptyResult: AnalyzeResponse | null = null;
const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

function App() {
  const [model, setModel] = useState("qwen-plus");
  const [researchQuestion, setResearchQuestion] = useState("");
  const [transcript, setTranscript] = useState("");
  const [result, setResult] = useState<AnalyzeResponse | null>(emptyResult);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  const summaryStats = useMemo(() => {
    if (!result) {
      return [
        { label: "访谈片段", value: "0", hint: "等待文本输入" },
        { label: "主题标签", value: "0", hint: "待分析" },
        { label: "聚类分组", value: "0", hint: "待分析" },
        { label: "情感片段", value: "0", hint: "待分析" },
      ];
    }

    const totalThemes = Object.values(result.all_codes).reduce((count, items) => count + items.length, 0);
    return [
      { label: "访谈片段", value: String(result.segments_count), hint: "已完成拆分" },
      { label: "主题标签", value: String(totalThemes), hint: "保留原文引述" },
      { label: "聚类分组", value: String(result.affinity.clusters.length), hint: "用于总结共性" },
      { label: "情感片段", value: String(result.sentiments.length), hint: "含整体情感判断" },
    ];
  }, [result]);

  async function handleAnalyze() {
    if (!transcript.trim()) {
      setError("请先输入访谈文本。");
      return;
    }

    setIsLoading(true);
    setError("");

    try {
      const requestUrl = apiBaseUrl ? `${apiBaseUrl}/api/analyze` : "/api/analyze";
      const response = await fetch(requestUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          transcript,
          model,
          research_question: researchQuestion,
        }),
      });

      const payload = (await response.json()) as AnalyzeResponse | { detail?: string };
      if (!response.ok) {
        throw new Error("detail" in payload ? payload.detail || "分析失败" : "分析失败");
      }

      setResult(payload as AnalyzeResponse);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "分析失败，请稍后重试。");
    } finally {
      setIsLoading(false);
    }
  }

  async function handleExport() {
    if (!result) {
      setError("请先完成一次分析。");
      return;
    }

    try {
      const requestUrl = apiBaseUrl ? `${apiBaseUrl}/api/export-report` : "/api/export-report";
      const response = await fetch(requestUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          all_codes: result.all_codes,
          sentiments: result.sentiments,
          affinity: result.affinity,
          insights: result.insights,
          research_question: researchQuestion,
        }),
      });
      const payload = (await response.json()) as { markdown?: string; detail?: string };
      if (!response.ok || !payload.markdown) {
        throw new Error(payload.detail || "导出失败");
      }

      const blob = new Blob([payload.markdown], { type: "text/markdown;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = "访谈分析报告.md";
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "导出失败，请稍后重试。");
    }
  }

  function handleReset() {
    setTranscript("");
    setResearchQuestion("");
    setResult(null);
    setError("");
  }

  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top_left,_rgba(190,214,255,0.45),_transparent_26%),linear-gradient(180deg,_#f6f8fc_0%,_#edf2f7_100%)] text-foreground">
      <div className="mx-auto flex min-h-screen w-full max-w-[1480px] flex-col gap-6 px-4 py-5 lg:px-6 lg:py-6">
        <section className="grid gap-4 rounded-[34px] border border-white/60 bg-slate-950 px-6 py-7 text-white shadow-panel lg:grid-cols-[1.5fr_1fr]">
          <div className="space-y-4">
            <Badge className="w-fit border-white/15 bg-white/10 text-white" variant="neutral">
              用户研究分析工作台
            </Badge>
            <div className="space-y-3">
              <h1 className="font-display text-3xl font-semibold tracking-tight lg:text-[2.8rem]">把访谈文本整理成可追溯的研究证据</h1>
              <p className="max-w-2xl text-sm leading-7 text-slate-300 lg:text-[15px]">
                这个面板围绕真实研究流程搭建：先粘贴文本，再快速跑通拆分、主题、情感、聚类与洞察，最后回到原文核对证据并导出报告。
              </p>
            </div>
          </div>
          <div className="grid gap-3 self-stretch sm:grid-cols-3 lg:grid-cols-1">
            {[
              ["证据优先", "所有结果都应能回到原话。"],
              ["处理高效", "配置、分析和导出都在一个工作台内完成。"],
              ["状态清楚", "空态、处理中、完成态都有明确反馈。"],
            ].map(([title, copy]) => (
              <div key={title} className="rounded-[24px] border border-white/10 bg-white/6 p-4">
                <div className="text-sm font-medium">{title}</div>
                <div className="mt-2 text-sm leading-6 text-slate-300">{copy}</div>
              </div>
            ))}
          </div>
        </section>

        <section className="grid gap-6 xl:grid-cols-[420px_minmax(0,1fr)]">
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>分析配置</CardTitle>
                <CardDescription>控制模型和本次研究目标。前端只暴露必要输入，不暴露密钥。</CardDescription>
              </CardHeader>
              <CardContent className="space-y-5">
                <div className="space-y-2">
                  <Label>模型选择</Label>
                  <Select value={model} onValueChange={setModel}>
                    <SelectTrigger>
                      <SelectValue placeholder="选择模型" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="qwen-turbo">qwen-turbo</SelectItem>
                      <SelectItem value="qwen-plus">qwen-plus</SelectItem>
                      <SelectItem value="qwen-max">qwen-max</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="research-question">研究目标</Label>
                  <Textarea
                    id="research-question"
                    className="min-h-28"
                    placeholder="例：用户为何在关键环节流失？"
                    value={researchQuestion}
                    onChange={(event) => setResearchQuestion(event.target.value)}
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>访谈文本</CardTitle>
                <CardDescription>支持标准 Q&A，也支持较长段落。建议保留原话，方便后续回溯。</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <Textarea
                  className="min-h-[380px]"
                  placeholder="粘贴访谈转录文本..."
                  value={transcript}
                  onChange={(event) => setTranscript(event.target.value)}
                />

                <div className="flex flex-wrap gap-3">
                  <Button type="button" variant="secondary" onClick={() => setTranscript(demoTranscript)}>
                    加载示例
                  </Button>
                  <Button type="button" variant="outline" onClick={handleReset}>
                    <RefreshCcw className="mr-2 h-4 w-4" />
                    清空内容
                  </Button>
                  <Button type="button" className="ml-auto" onClick={handleAnalyze} disabled={isLoading}>
                    {isLoading ? <LoaderCircle className="mr-2 h-4 w-4 animate-spin" /> : <Sparkles className="mr-2 h-4 w-4" />}
                    {isLoading ? "分析中..." : "开始分析"}
                  </Button>
                </div>

                {error ? <div className="rounded-2xl bg-rose-50 px-4 py-3 text-sm text-rose-700">{error}</div> : null}
              </CardContent>
            </Card>
          </div>

          <div className="space-y-6">
            <Card className="overflow-hidden">
              <CardHeader className="border-b border-border/70 bg-slate-50/70">
                <div className="flex flex-wrap items-start justify-between gap-4">
                  <div className="space-y-2">
                    <CardTitle>结果总览</CardTitle>
                    <CardDescription>先扫摘要，再进入编码、情感、聚类和洞察模块。</CardDescription>
                  </div>
                  <div className="flex items-center gap-3">
                    <Badge>{result?.model || model}</Badge>
                    <Button type="button" variant="outline" onClick={handleExport} disabled={!result}>
                      <Download className="mr-2 h-4 w-4" />
                      导出报告
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-6 pt-6">
                <div className="grid gap-3 md:grid-cols-2 2xl:grid-cols-4">
                  {summaryStats.map((stat) => (
                    <div key={stat.label} className="rounded-[24px] border border-slate-200 bg-slate-50/80 p-4">
                      <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">{stat.label}</div>
                      <div className="mt-3 font-display text-3xl font-semibold text-slate-950">{stat.value}</div>
                      <div className="mt-2 text-sm leading-6 text-slate-500">{stat.hint}</div>
                    </div>
                  ))}
                </div>

                <div className="flex flex-wrap gap-2">
                  {["文本预处理", "主题编码", "情感分析", "亲和图聚类", "洞察生成"].map((step, index) => (
                    <Badge key={step} variant={result ? "default" : index === 0 ? "warning" : "neutral"}>
                      {step}
                    </Badge>
                  ))}
                </div>

                <div className="rounded-[26px] border border-dashed border-slate-300 bg-white/70 p-5">
                  <div className="text-sm font-medium text-slate-900">研究摘要</div>
                  <p className="mt-2 text-sm leading-7 text-slate-600">
                    {result
                      ? "分析结果已生成。请优先查看高频主题与代表引述，再对照情感和聚类确认研究结论。"
                      : "等待分析。完成后这里会展示本轮结果的关键提示，帮助你快速决定先看哪一块。"}
                  </p>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>结果明细</CardTitle>
                <CardDescription>按模块查看细节，长访谈也能先总览再钻取。</CardDescription>
              </CardHeader>
              <CardContent>
                <Accordion type="multiple" defaultValue={["codes"]} className="space-y-3">
                  <AccordionItem value="codes">
                    <AccordionTrigger>01 主题编码</AccordionTrigger>
                    <AccordionContent>
                      {result ? (
                        <div className="space-y-4">
                          {Object.entries(result.all_codes).map(([segmentId, items]) => (
                            <div key={segmentId} className="rounded-[20px] border border-slate-200 bg-slate-50/70 p-4">
                              <div className="mb-3 flex items-center justify-between gap-3">
                                <div className="text-sm font-semibold text-slate-900">{segmentId}</div>
                                <Badge variant="neutral">{items.length} 条</Badge>
                              </div>
                              <div className="space-y-3">
                                {items.map((item, index) => (
                                  <div key={`${segmentId}-${index}`} className="rounded-[18px] bg-white p-4">
                                    <div className="text-sm font-semibold text-slate-900">{item.theme}</div>
                                    <p className="mt-2 text-sm leading-7 text-slate-600">“{item.quote}”</p>
                                    <p className="mt-2 text-sm leading-6 text-slate-500">{item.note}</p>
                                  </div>
                                ))}
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <EmptyState copy="分析完成后，这里会按片段展示主题标签、原文引述和研究员备注。" />
                      )}
                    </AccordionContent>
                  </AccordionItem>

                  <AccordionItem value="sentiment">
                    <AccordionTrigger>02 情感分析</AccordionTrigger>
                    <AccordionContent>
                      {result ? (
                        <div className="space-y-4">
                          {result.sentiments.map((item) => (
                            <div key={item.id} className="rounded-[20px] border border-slate-200 bg-slate-50/70 p-4">
                              <div className="mb-3 flex items-center justify-between gap-3">
                                <div className="text-sm font-semibold text-slate-900">{item.id}</div>
                                <Badge variant={item.overall === "负面" ? "danger" : item.overall === "正面" ? "success" : "neutral"}>
                                  {item.overall}
                                </Badge>
                              </div>
                              <div className="space-y-3">
                                {item.details.length ? (
                                  item.details.map((detail, index) => (
                                    <div key={`${item.id}-${index}`} className="rounded-[18px] bg-white p-4">
                                      <div className="flex flex-wrap items-center gap-2">
                                        <span className="text-sm font-semibold text-slate-900">{detail.aspect}</span>
                                        <Badge variant={detail.sentiment === "负面" ? "danger" : detail.sentiment === "正面" ? "success" : "neutral"}>
                                          {detail.sentiment}
                                        </Badge>
                                        <span className="text-xs text-slate-500">强度 {detail.intensity}</span>
                                      </div>
                                      <p className="mt-3 text-sm leading-7 text-slate-600">“{detail.evidence}”</p>
                                    </div>
                                  ))
                                ) : (
                                  <div className="rounded-[18px] bg-white p-4 text-sm text-slate-500">当前片段没有更细的情感拆解。</div>
                                )}
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <EmptyState copy="分析完成后，这里会展示整体情感和细分方面。" />
                      )}
                    </AccordionContent>
                  </AccordionItem>

                  <AccordionItem value="affinity">
                    <AccordionTrigger>03 亲和图聚类</AccordionTrigger>
                    <AccordionContent>
                      {result ? (
                        <div className="grid gap-4 xl:grid-cols-2">
                          {result.affinity.clusters.map((cluster, index) => (
                            <div key={`${cluster.group_name}-${index}`} className="rounded-[20px] border border-slate-200 bg-slate-50/70 p-5">
                              <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">分组 {String(index + 1).padStart(2, "0")}</div>
                              <div className="mt-2 text-base font-semibold text-slate-900">{cluster.group_name}</div>
                              <p className="mt-3 text-sm leading-7 text-slate-600">{cluster.description}</p>
                              <div className="mt-4 flex flex-wrap gap-2">
                                {cluster.themes.map((theme) => (
                                  <Badge key={theme} variant="neutral">
                                    {theme}
                                  </Badge>
                                ))}
                              </div>
                              <blockquote className="mt-4 rounded-[18px] bg-white px-4 py-3 text-sm leading-7 text-slate-600">
                                “{cluster.representative_quote}”
                              </blockquote>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <EmptyState copy="分析完成后，这里会汇总主题并形成聚类卡片。" />
                      )}
                    </AccordionContent>
                  </AccordionItem>

                  <AccordionItem value="insight">
                    <AccordionTrigger>04 关键洞察</AccordionTrigger>
                    <AccordionContent>
                      {result ? (
                        <article className="rounded-[20px] border border-slate-200 bg-slate-50/70 p-5">
                          <pre className="whitespace-pre-wrap font-sans text-sm leading-7 text-slate-700">{result.insights}</pre>
                        </article>
                      ) : (
                        <EmptyState copy="分析完成后，这里会生成研究洞察和后续建议。" />
                      )}
                    </AccordionContent>
                  </AccordionItem>
                </Accordion>
              </CardContent>
            </Card>
          </div>
        </section>
      </div>
    </main>
  );
}

function EmptyState({ copy }: { copy: string }) {
  return (
    <div className="rounded-[20px] border border-dashed border-slate-300 bg-slate-50/70 px-5 py-6 text-sm leading-7 text-slate-500">
      {copy}
    </div>
  );
}

export default App;
