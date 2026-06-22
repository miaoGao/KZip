"""KnowledgeZip: question-aware knowledge compression for multi-hop QA.

A minimal, inference-only reference implementation. See README.md.
"""

from .client import LLM
from .pipeline import answer, build_context, compress, run_question

__all__ = ["LLM", "run_question", "compress", "answer", "build_context"]
