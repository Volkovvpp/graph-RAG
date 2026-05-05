from typing import List, Union
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

class DocumentLoader:
    """Helper class to load documents of different types."""

    @staticmethod
    def load_file(file_path: Union[str, Path]) -> List[Document]:
        """Load a single file based on its extension.

        Args:
            file_path: The path to the file to load.

        Returns:
            A list of LangChain Document objects.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        extension = path.suffix.lower()

        if extension == ".pdf":
            loader = PyPDFLoader(str(path))
        elif extension in [".txt", ".md"]:
            loader = TextLoader(str(path), encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file extension: {extension}")

        return loader.load()

    @staticmethod
    def load_directory(directory_path: Union[str, Path], extensions: List[str] = [".pdf", ".txt", ".md"]) -> List[Document]:
        """Load all supported documents from a directory.

        Args:
            directory_path: The path to the directory.
            extensions: A list of file extensions to include.

        Returns:
            A list of LangChain Document objects from all loaded files.
        """
        path = Path(directory_path)
        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")

        documents = []
        for ext in extensions:
            for file in path.glob(f"*{ext}"):
                try:
                    documents.extend(DocumentLoader.load_file(file))
                except Exception as e:
                    print(f"Error loading {file}: {e}")

        return documents

