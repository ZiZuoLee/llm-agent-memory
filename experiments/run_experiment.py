"""Simple experiment runner for LLM memory modes."""

from __future__ import annotations

from typing import Dict, List

from llm.llm_client import LLMClient
from memory.context_memory import ContextMemory
from memory.memory_router import MemoryRouter
from memory.profile_memory import ProfileMemory
from memory.retrieval_memory import RetrievalMemory
from prompt.prompt_builder import PromptBuilder


def _log_turn(turn_index: int, user_text: str, assistant_text: str) -> None:
    """Print a formatted turn to the console."""
    print(f"\nTurn {turn_index}")
    print(f"User: {user_text}")
    print(f"Assistant: {assistant_text}")


def _build_messages_no_memory(query: str) -> List[Dict[str, str]]:
    """Build messages without any memory."""
    return [{"role": "user", "content": query}]


def _build_messages_with_context(
    builder: PromptBuilder,
    context_memory: ContextMemory,
    query: str,
) -> List[Dict[str, str]]:
    """Build messages using context memory only."""
    return builder.build_messages(
        context_memories=context_memory.get_context(),
        retrieved_memories=[],
        profile_prompt="",
        user_query=query,
    )


def _build_messages_with_retrieval(
    builder: PromptBuilder,
    retrieval_memory: RetrievalMemory,
    query: str,
    k: int = 3,
) -> List[Dict[str, str]]:
    """Build messages using retrieval memory only."""
    retrieved = retrieval_memory.retrieve(query, k)
    return builder.build_messages(
        context_memories=[],
        retrieved_memories=retrieved,
        profile_prompt="",
        user_query=query,
    )


def _build_messages_hierarchical(
    builder: PromptBuilder,
    router: MemoryRouter,
    query: str,
) -> List[Dict[str, str]]:
    """Build messages using combined memories."""
    memories = router.collect_memories(query)
    return builder.build_messages(
        context_memories=memories["context"],
        retrieved_memories=memories["retrieval"],
        profile_prompt=memories["profile"],
        user_query=query,
    )


def run_experiment(mode: str, user_queries: List[str]) -> None:
    """Run a multi-turn simulation for a given memory mode."""
    llm = LLMClient()
    builder = PromptBuilder()
    context_memory = ContextMemory(max_turns=4)
    retrieval_memory = RetrievalMemory()
    profile_memory = ProfileMemory()
    router = MemoryRouter(
        context_memory=context_memory,
        retrieval_memory=retrieval_memory,
        profile_memory=profile_memory,
        retrieval_k=3,
    )

    for index, query in enumerate(user_queries, start=1):
        if mode == "no_memory":
            messages = _build_messages_no_memory(query)
        elif mode == "context":
            messages = _build_messages_with_context(builder, context_memory, query)
        elif mode == "retrieval":
            messages = _build_messages_with_retrieval(builder, retrieval_memory, query)
        elif mode == "hierarchical":
            messages = _build_messages_hierarchical(builder, router, query)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        assistant_text = llm.generate(messages)
        _log_turn(index, query, assistant_text)

        context_memory.add_turn(query, assistant_text)
        retrieval_memory.add_memory(f"User: {query}\nAssistant: {assistant_text}")
        profile_memory.update_from_text(query)


def main() -> None:
    """Entry point for running a demo experiment."""
    mode = "hierarchical"
    user_queries = [
        "Hi, I like concise answers.",
        "Remind me what I said about answer style.",
        "Give me a short summary of our chat.",
    ]
    print(f"Running experiment in mode: {mode}")
    run_experiment(mode, user_queries)


if __name__ == "__main__":
    main()
