from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from typing import Iterable

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_pdfs(uploaded_files: Iterable, chunk_size: int = 1400, chunk_overlap: int = 150) -> tuple[list[Document], str]:
    """
    Extract PDF pages, retain source/page metadata, and split text into chunks.
    chunk_size/overlap are character-based approximations, not exact token counts.
    """
    page_documents: list[Document] = []
    digest = sha256()

    for uploaded in uploaded_files:
        raw = uploaded.getvalue()
        digest.update(uploaded.name.encode("utf-8", errors="ignore"))
        digest.update(raw)
        try:
            reader = PdfReader(BytesIO(raw))
        except Exception as exc:
            raise ValueError(f"Could not open '{uploaded.name}' as a PDF: {exc}") from exc

        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                page_documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": uploaded.name, "page": page_number},
                    )
                )

    if not page_documents:
        raise ValueError(
            "No selectable text was found in the uploaded PDFs. "
            "If these are scanned documents, OCR is required; OCR is not included in this prototype."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(page_documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index
        # Add a stable per-chunk ID so repeated indexing is deterministic.
        chunk.metadata["id"] = sha256(
            f'{chunk.metadata.get("source")}:{chunk.metadata.get("page")}:{chunk.metadata.get("start_index")}:{chunk.page_content}'.encode("utf-8")
        ).hexdigest()

    return chunks, digest.hexdigest()[:16]
