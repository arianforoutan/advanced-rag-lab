"""Load the markdown knowledge base and split it into retrievable chunks."""

from __future__ import annotations

import glob
import os

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config


def load_documents(knowledge_base_dir=config.KNOWLEDGE_BASE_DIR) -> list[Document]:
    """Load every markdown file, tagging each with a ``doc_type`` from its folder.

    Each top-level folder under the knowledge base (company, products,
    contracts, employees) becomes the ``doc_type`` metadata used later for
    access filtering.
    """
    documents: list[Document] = []
    for folder in glob.glob(os.path.join(str(knowledge_base_dir), "*")):
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(
            folder,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        for document in loader.load():
            document.metadata["doc_type"] = doc_type
            documents.append(document)
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks. BM25 and Chroma both use these."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def filter_public_chunks(
    chunks: list[Document],
    allowed_doc_types: list[str] = config.ALLOWED_DOC_TYPES,
) -> list[Document]:
    """Keep only chunks whose ``doc_type`` is allowed by the retrieval policy."""
    return [
        chunk
        for chunk in chunks
        if chunk.metadata.get("doc_type") in allowed_doc_types
    ]
