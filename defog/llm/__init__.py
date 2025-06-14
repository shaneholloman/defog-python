"""LLM module with memory management capabilities."""

from .utils import chat_async, LLMResponse
from .utils_memory import (
    chat_async_with_memory,
    create_memory_manager,
    MemoryConfig,
)
from .memory import (
    MemoryManager,
    ConversationHistory,
    compactify_messages,
    TokenCounter,
)
from .pdf_processor import analyze_pdf, PDFAnalysisInput, ClaudePDFProcessor
from .pdf_data_extractor import PDFDataExtractor, extract_pdf_data

__all__ = [
    # Core functions
    "chat_async",
    "chat_async_with_memory",
    "LLMResponse",
    # Memory management
    "MemoryManager",
    "ConversationHistory",
    "MemoryConfig",
    "create_memory_manager",
    "compactify_messages",
    "TokenCounter",
    # PDF processing
    "analyze_pdf",
    "PDFAnalysisInput",
    "ClaudePDFProcessor",
    "PDFDataExtractor",
    "extract_pdf_data",
]
