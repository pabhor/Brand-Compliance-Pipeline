import json
import os
import logging
import re
from typing import Dict, Any

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
from langchain_core.messages import SystemMessage, HumanMessage

# Import the State schema
from backend.src.graph.state import VideoAuditState

# Import the Service
from backend.src.services.video_indexer import VideoIndexerService

# Configure Logger
logger = logging.getLogger("brand-guardian")
logging.basicConfig(level=logging.INFO)


# --- NODE 1: THE INDEXER ---
def index_video_node(state: VideoAuditState) -> Dict[str, Any]:
    """
    Downloads YouTube video, uploads to Azure VI, and extracts insights.
    """
    video_url = state.get("video_url")
    video_id_input = state.get("video_id", "vid_demo")

    logger.info(f"--- [Node: Indexer] Processing: {video_url} ---")

    local_filename = "temp_audit_video.mp4"

    try:
        vi_service = VideoIndexerService()

        # 1. DOWNLOAD
        if video_url and ("youtube.com" in video_url or "youtu.be" in video_url):
            local_path = vi_service.download_youtube_video(video_url, output_path=local_filename)
        else:
            raise Exception("Please provide a valid YouTube URL for this test.")

        # 2. UPLOAD
        azure_video_id = vi_service.upload_video(local_path, video_name=video_id_input)
        logger.info(f"Upload Success. Azure ID: {azure_video_id}")

        # 3. CLEANUP
        if os.path.exists(local_path):
            os.remove(local_path)

        # 4. WAIT
        raw_insights = vi_service.wait_for_processing(azure_video_id)

        # 5. EXTRACT
        clean_data = vi_service.extract_data(raw_insights)

        logger.info("--- [Node: Indexer] Extraction Complete ---")
        return clean_data

    except Exception as e:
        logger.error(f"Video Indexer Failed: {e}")
        return {
            "errors": [str(e)],
            "final_status": "FAIL",
            "transcript": "",
            "ocr_text": [],
        }


# --- NODE 2: THE COMPLIANCE AUDITOR ---
def audit_content_node(state: VideoAuditState) -> Dict[str, Any]:
    """
    Performs Retrieval-Augmented Generation (RAG) to audit the content.
    """
    logger.info("--- [Node: Auditor] querying Knowledge Base & LLM ---")

    transcript = state.get("transcript", "")
    if not transcript:
        logger.warning("No transcript available. Skipping Audit.")
        return {
            "final_status": "FAIL",
            "final_report": "Audit skipped because video processing failed (No Transcript).",
        }

    # ---- ENV (debug) ----
    aoai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    aoai_key = os.getenv("AZURE_OPENAI_API_KEY")
    aoai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    aoai_chat_deploy = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
    aoai_embed_deploy = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

    logger.info("AOAI endpoint=%r", aoai_endpoint)
    logger.info("AOAI api_version=%r", aoai_api_version)
    logger.info("AOAI chat deployment=%r", aoai_chat_deploy)
    logger.info("AOAI embedding deployment=%r", aoai_embed_deploy)

    # Basic validation (fail fast with a clear error)
    missing = []
    for k, v in [
        ("AZURE_OPENAI_ENDPOINT", aoai_endpoint),
        ("AZURE_OPENAI_API_KEY", aoai_key),
        ("AZURE_OPENAI_API_VERSION", aoai_api_version),
        ("AZURE_OPENAI_CHAT_DEPLOYMENT", aoai_chat_deploy),
        ("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", aoai_embed_deploy),
        ("AZURE_SEARCH_ENDPOINT", os.getenv("AZURE_SEARCH_ENDPOINT")),
        ("AZURE_SEARCH_API_KEY", os.getenv("AZURE_SEARCH_API_KEY")),
        ("AZURE_SEARCH_INDEX_NAME", os.getenv("AZURE_SEARCH_INDEX_NAME")),
    ]:
        if not v:
            missing.append(k)

    if missing:
        msg = f"Missing required environment variables: {', '.join(missing)}"
        logger.error(msg)
        return {"errors": [msg], "final_status": "FAIL"}

    # Initialize Clients (explicit endpoint/key prevents silent config issues)
    llm = AzureChatOpenAI(
        azure_deployment=aoai_chat_deploy,
        api_version=aoai_api_version,
        azure_endpoint=aoai_endpoint,
        api_key=aoai_key,
        temperature=0.0,
    )

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=aoai_embed_deploy,
        api_version=aoai_api_version,
        azure_endpoint=aoai_endpoint,
        api_key=aoai_key,
        check_embedding_ctx_length=False,  # prevents token-array path
    )

    vector_store = AzureSearch(
        azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        azure_search_key=os.getenv("AZURE_SEARCH_API_KEY"),
        index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
        embedding_function=embeddings.embed_query,
    )

    # RAG Retrieval
    ocr_text = state.get("ocr_text", [])
    query_text = f"{transcript} {' '.join(ocr_text)}"
    docs = vector_store.similarity_search(query_text, k=3)

    retrieved_rules = "\n\n".join([doc.page_content for doc in docs])

    system_prompt = f"""
You are a Senior Brand Compliance Auditor.

OFFICIAL REGULATORY RULES:
{retrieved_rules}

INSTRUCTIONS:
1. Analyze the Transcript and OCR text below.
2. Identify ANY violations of the rules.
3. Return strictly JSON in the following format:

{{
  "compliance_results": [
    {{
      "category": "Claim Validation",
      "severity": "CRITICAL",
      "description": "Explanation of the violation..."
    }}
  ],
  "status": "FAIL",
  "final_report": "Summary of findings..."
}}

If no violations are found, set "status" to "PASS" and "compliance_results" to [].
""".strip()

    user_message = f"""
VIDEO METADATA: {state.get('video_metadata', {})}
TRANSCRIPT: {transcript}
ON-SCREEN TEXT (OCR): {ocr_text}
""".strip()

    try:
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message),
            ]
        )

        content = response.content or ""

        # Clean Markdown fenced code blocks if present
        if "```" in content:
            m = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
            if m:
                content = m.group(1)

        audit_data = json.loads(content.strip())

        return {
            "compliance_results": audit_data.get("compliance_results", []),
            "final_status": audit_data.get("status", "FAIL"),
            "final_report": audit_data.get("final_report", "No report generated."),
        }

    except Exception as e:
        logger.error(f"System Error in Auditor Node: {str(e)}")
        logger.error(f"Raw LLM Response: {response.content if 'response' in locals() else 'None'}")
        return {
            "errors": [str(e)],
            "final_status": "FAIL",
        }