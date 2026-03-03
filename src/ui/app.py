import streamlit as st
import asyncio
import tempfile
import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.hybrid import HybridRetriever
from src.generation.synthesizer import Synthesizer
from src.core.logger import get_logger

logger = get_logger("ui")

st.set_page_config(page_title="Graph-RAG Knowledge Assistant", layout="wide")

st.title("🕸️ Graph-RAG Knowledge Assistant")
st.markdown("Upload documents to build a knowledge graph and ask questions about them.")

# --- Sidebar: Document Ingestion ---
st.sidebar.header("📂 Document Ingestion")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF, TXT, or MD files",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True
)

process_button = st.sidebar.button("Process Documents")

# --- Main Area: Q&A ---
st.header("💬 Ask a Question")
query = st.text_input("Enter your question here:")
search_button = st.button("Search & Answer")

# --- Helper Functions ---

async def run_ingestion(files):
    pipeline = IngestionPipeline()
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()

    total_files = len(files)

    try:
        # Create a temporary directory to store uploaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            for i, uploaded_file in enumerate(files):
                status_text.text(f"Processing {uploaded_file.name}...")

                # Save file to temp path
                file_path = temp_path / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Run pipeline on this file
                await pipeline.run(file_path)

                # Update progress
                progress = (i + 1) / total_files
                progress_bar.progress(progress)

            status_text.text("✅ Ingestion Complete!")
            st.sidebar.success(f"Successfully processed {total_files} files.")

    except Exception as e:
        status_text.text("❌ Error during ingestion")
        st.sidebar.error(f"Error: {str(e)}")
        logger.error(f"Ingestion error: {e}")
    finally:
        # Close connections if needed (though usually kept open for query)
        pass

async def get_answer(user_query):
    retriever = HybridRetriever()
    synthesizer = Synthesizer()

    with st.spinner("Retrieving context and generating answer..."):
        try:
            # 1. Retrieve Context
            retrieval_result = await retriever.retrieve(user_query)
            context = retrieval_result.get("context", "")
            sources = retrieval_result.get("sources", [])

            if not context:
                return "I couldn't find any relevant information in the uploaded documents.", []

            # 2. Generate Answer
            answer = await synthesizer.generate_response(user_query, context)

            return answer, sources

        except Exception as e:
            logger.error(f"Query error: {e}")
            return f"An error occurred: {str(e)}", []

# --- Event Handlers ---

async def handle_ingestion():
    if uploaded_files:
        await run_ingestion(uploaded_files)
    else:
        st.sidebar.warning("Please upload files first.")

async def handle_query():
    if query:
        answer, sources = await get_answer(query)

        st.markdown("### Answer")
        st.write(answer)

        if sources:
            with st.expander("View Sources"):
                for src in sources:
                    st.markdown(f"- **{src.get('chunk_id', 'Unknown')}**: {src.get('text', '')[:200]}...")
    else:
        st.warning("Please enter a question.")

if process_button:
    asyncio.run(handle_ingestion())

if search_button:
    asyncio.run(handle_query())

# streamlit run D:\graph-RAG\src\ui\app.py
