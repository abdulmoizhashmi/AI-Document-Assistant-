# AI Document Assistant (RAG)

A local Streamlit application for uploading PDFs, indexing their text with embeddings in ChromaDB, retrieving relevant passages, reranking them, and answering questions with page citations.

## Try it online

The [GitHub Pages project site](https://abdulmoizhashmi.github.io/AI-Document-Assistant-/) describes the app and links to its deployment flow. GitHub Pages only hosts static files; the PDF upload, indexing, and AI chat run in the Python Streamlit app.

To host a working instance:
1. Open [the Streamlit deployment page for this repository](https://share.streamlit.io/deploy?repository=abdulmoizhashmi%2FAI-Document-Assistant-&branch=main&mainModule=app.py) and sign in to Streamlit Community Cloud.
2. Select the `main` branch and `app.py` as the app's main file, then deploy.
3. In the deployed app's **Settings → Secrets**, add at least one provider key:
   ```toml
   GROQ_API_KEY = "your-groq-api-key"
   # Or:
   OPENROUTER_API_KEY = "your-openrouter-api-key"
   ```
4. Save the secrets and let Streamlit restart the app. Keep provider keys in Streamlit Secrets; never commit them to GitHub.

## Features
- Multiple PDF upload
- PDF text extraction with page metadata
- Recursive text splitting (target ~512 tokens; configured using character lengths as an approximation, overlap ~50 tokens)
- Sentence Transformers embeddings: `all-MiniLM-L6-v2`
- Persistent local ChromaDB vector store
- Similarity retrieval plus cross-encoder reranking
- Grounded answers with document/page citations
- LLM failover: Groq first, then OpenRouter Free Models Router
- Helpful error messages when no API keys are configured or providers are unavailable

## Requirements
- Windows 10/11
- Python 3.10–3.12 recommended
- Internet connection for package/model downloads and hosted API calls
- At least one API key for hosted LLM responses. Free plans have limits and availability can change.

## Setup (VS Code / PowerShell)
1. Extract this ZIP.
2. Open the extracted `ai_document_assistant` folder in VS Code.
3. In the terminal, run:
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env`:
   ```powershell
   Copy-Item .env.example .env
   ```
5. Open `.env` and add at least one key:
   - Groq: https://console.groq.com/keys
   - OpenRouter: https://openrouter.ai/settings/keys
6. Start the app:
   ```powershell
   streamlit run app.py
   ```
7. Open the local URL Streamlit prints (normally `http://localhost:8501`).

You can also double-click `run_windows.bat`; it creates the virtual environment, installs dependencies, and opens `.env` if it does not exist.

## LLM fallback behaviour
- The app tries Groq first if `GROQ_API_KEY` is configured.
- If Groq fails (for example, rate limit, temporary outage, invalid model, or network error), the app tries OpenRouter if `OPENROUTER_API_KEY` is configured.
- OpenRouter uses `openrouter/free`, which chooses among currently available free models. Its model pool, quotas, and limits may change.
- If neither provider succeeds, the app shows a safe error rather than fabricating an answer.
- The application-level fallback is not a guarantee of uninterrupted service; both providers require working keys, network access, and available quotas.

## Project structure
```text
ai_document_assistant/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── run_windows.bat
├── README.md
├── src/
│   ├── __init__.py
│   ├── document_loader.py
│   ├── vector_store.py
│   ├── retriever.py
│   └── answer_generator.py
└── data/
    └── chroma_db/
```

## Notes
- The first run downloads embedding and reranker models; this may take time and disk space.
- PDFs must contain extractable text. Scanned/image-only PDFs need OCR, which is not included.
- Text splitting uses character lengths because common recursive splitters count characters by default. The values approximate the requested 512-token chunks and 50-token overlap; they are not guaranteed exact token counts.
- Source page numbers are stored as 1-based PDF page numbers.
- This is a learning prototype, not a production-secure multi-user service.
