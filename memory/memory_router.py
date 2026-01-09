"""Memory router module for aggregating different memory types."""

from __future__ import annotations

from typing import Dict, List

from memory.context_memory import ContextMemory
from memory.profile_memory import ProfileMemory
from memory.retrieval_memory import RetrievalMemory


class MemoryRouter:
    """Combine context, retrieval, and profile memories for a query."""

    def __init__(
        self,
        context_memory: ContextMemory,
        retrieval_memory: RetrievalMemory,
        profile_memory: ProfileMemory,
        retrieval_k: int = 3,
    ) -> None:
        """Initialize with memory components and retrieval size."""
        self._context_memory = context_memory
        self._retrieval_memory = retrieval_memory
        self._profile_memory = profile_memory
        self._retrieval_k = retrieval_k

    def collect_memories(self, query: str) -> Dict[str, List[str] | str | list[dict]]:
        """Collect memories for a query in a deterministic structure."""
        context = self._context_memory.get_context()
        retrieval = self._retrieval_memory.retrieve(query, self._retrieval_k)
        profile = self._profile_memory.get_profile_prompt()
        return {"context": context, "retrieval": retrieval, "profile": profile}
