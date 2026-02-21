# 🎥 Compliance QA Pipeline  
AI-Powered Video Compliance Auditing System using Azure & LLMs

---

## 🚀 Overview

The Compliance QA Pipeline is an AI-driven system that automatically audits video content for regulatory and policy compliance.

It integrates:

- 🎬 Azure Video Indexer (Transcript Extraction)
- 🤖 Azure OpenAI (LLM-based reasoning)
- 🔎 Knowledge Base + Retrieval
- 🧠 LangGraph workflow orchestration
- ⚡ FastAPI backend
- 📊 Structured compliance reporting

The system processes a YouTube video, extracts transcripts, analyzes content using an LLM, and generates a structured compliance audit report.

---

## 🏗 Architecture

![Compliance QA Architecture](screenshot/architecture.png)


## 🔬 Methodology Operations : 

- The Compliance QA Pipeline follows a modular, graph-based orchestration approach for AI-driven content auditing. The system begins by ingesting a video URL and leveraging Azure Video Indexer to extract structured metadata and speech transcripts. This transcript serves as the primary input for downstream analysis. 

- A LangGraph workflow orchestrates the execution pipeline, managing state transitions between indexing and auditing nodes while ensuring deterministic and reproducible processing. 

- For reasoning and compliance evaluation, Azure OpenAI is used to perform structured analysis against policy constraints. 

- LangSmith is integrated for tracing, observability, and debugging of LLM interactions, enabling transparent monitoring of prompt execution, token usage, and model behavior. 

- This layered architecture separates ingestion, orchestration, reasoning, and reporting components, making the system scalable, auditable, and production-ready for enterprise compliance workflows.


## 📌 Future Improvements

- Add vector database for policy retrieval

- Add Streamlit UI

- Add batch video auditing

