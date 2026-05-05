# User Interview Analysis

一个面向用户研究场景的访谈文本自动分析工作台，用于把原始访谈记录快速整理为可追溯的研究证据、结构化洞察和可导出的分析报告。

## 项目简介

这个项目围绕真实用研流程搭建，而不是只做一次性的摘要生成。输入访谈转录文本后，系统会按固定分析链路完成拆分、编码、情感判断、聚类归纳和洞察生成，并尽量保留原文证据，方便研究人员回看与校验。

仓库当前包含两条使用路径：

- `FastAPI + React`：前后端分离的工作台版本，适合演示完整交互流程。
- `Gradio`：单文件原型版本，适合快速验证分析链路和 Prompt 效果。

## 核心能力

- 文本预处理：支持标准 Q&A 访谈文本，也支持较自由的长段落输入
- 主题编码：为每段访谈提取主题标签、原文引述和备注信息
- 情感分析：输出整体情感及更细粒度的方面级判断
- 亲和图聚类：将主题标签进一步归纳为分组卡片
- 洞察生成：结合研究目标生成可执行的研究发现
- 报告导出：支持导出 Markdown 格式的分析报告
- 研究目标注入：在分析阶段显式带入本次研究问题
- 多访谈对比：支持最多 3 份访谈的横向综合分析

## 技术栈

- 后端：`Python`、`FastAPI`、`Pydantic`
- AI 调用：`OpenAI SDK`、阿里云百炼兼容接口
- 原型界面：`Gradio`
- 前端：`React`、`TypeScript`、`Vite`、`Tailwind CSS`
- 测试：`pytest`、`fastapi.testclient`

## 项目结构

```text
.
├─ backend_api.py                  # FastAPI 接口层
├─ interview_analysis_app.py       # Gradio 原型与核心分析逻辑
├─ tests/                          # 核心逻辑与接口测试
├─ frontend/                       # React + Vite 前端工作台
└─ docs/                           # 方案与实现文档
```

## 本地运行

### 1. 环境准备

先在项目根目录创建 `.env`，填写百炼 API Key：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

### 2. 安装后端依赖

仓库当前未提供独立的 `requirements.txt`，可以先按下面命令安装最小依赖集：

```bash
pip install fastapi uvicorn gradio openai python-dotenv tenacity pydantic pytest httpx
```

### 3. 启动 FastAPI 后端

```bash
uvicorn backend_api:app --host 127.0.0.1 --port 8000 --reload
```

接口说明：

- `GET /health`：健康检查
- `POST /api/analyze`：提交访谈文本并返回结构化分析结果
- `POST /api/export-report`：导出 Markdown 报告

### 4. 启动前端工作台

先进入前端目录并安装依赖：

```bash
cd frontend
npm install
```

确认 `frontend/.env` 中的接口地址指向本地后端，例如：

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

然后启动前端：

```bash
npm run dev
```

默认访问地址通常为：

```text
http://127.0.0.1:5173
```

### 5. 启动 Gradio 原型

如果只想快速验证单文件原型，可以直接运行：

```bash
python interview_analysis_app.py
```

默认会在本地启动：

```text
http://127.0.0.1:7860
```

## 测试

运行全部测试：

```bash
pytest tests/test_api.py tests/test_core.py
```

测试目前覆盖的重点包括：

- API Key 不再硬编码
- JSON 解析的健壮性
- 研究目标注入逻辑
- 报告导出结构
- FastAPI 接口返回结构

## 截图说明

为了保持仓库精简，当前公开仓库未提交界面截图文件。若用于作品展示、申请材料或项目介绍，建议补充以下截图：

- 工作台首页：展示研究目标输入、模型选择和访谈文本输入区
- 结果总览：展示访谈片段数、主题标签数、聚类数和情感片段数
- 结果明细：展示主题编码、情感分析、亲和图聚类和关键洞察模块
- 多访谈对比：展示多份访谈横向分析结果

你本地已存在可用截图文件：

- `ui-review.png`
- `ui-review-desktop.png`

如果后续要在仓库首页直接展示，推荐将截图移动到 `docs/assets/` 后再在 README 中引用。

## 适用场景

- 用户访谈整理
- 可用性研究初步归纳
- 产品需求探索阶段的质性分析
- AI 用研工作流原型演示

## 当前边界

- 当前主要面向中文访谈文本
- 多访谈对比上限为 3 份，偏向演示与轻量研究场景
- 输出质量仍依赖输入文本质量与模型表现，建议保留人工复核

