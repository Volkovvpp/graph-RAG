from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.core.config import settings

class TextSplitter:
    """Helper class to split text into chunks."""

    _splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

    @staticmethod
    def split_documents(documents: List[Document]) -> List[Document]:
        """Split a list of documents into smaller chunks.

        Args:
            documents: A list of LangChain Document objects.

        Returns:
            A list of smaller LangChain Document objects (chunks).
        """
        return TextSplitter._splitter.split_documents(documents)

    @staticmethod
    def split_text(text: str) -> List[str]:
        """Split a raw string into chunks.

        Args:
            text: The text to split.

        Returns:
            A list of string chunks.
        """
        return TextSplitter._splitter.split_text(text)

