"""Simple experiment runner for LLM memory modes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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


def run_experiment(mode: str, user_queries: List[str]) -> List[str]:
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

    assistant_responses: List[str] = []
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
        assistant_responses.append(assistant_text)

        # Memory isolation: update only the components relevant to the active mode.
        if mode == "context":
            context_memory.add_turn(query, assistant_text)
        elif mode == "retrieval":
            retrieval_memory.add_memory(f"User: {query}\nAssistant: {assistant_text}")
        elif mode == "hierarchical":
            context_memory.add_turn(query, assistant_text)
            retrieval_memory.add_memory(f"User: {query}\nAssistant: {assistant_text}")
            profile_memory.update_from_text(query)
    return assistant_responses


def _scenario_short_preference() -> List[str]:
    """Create a short preference scenario."""
    return [
        "Hi, I like concise answers.",
        "Remind me what I said about answer style.",
        "Give me a short summary of our chat.",
    ]


def _scenario_long_preference() -> List[str]:
    """Create a long preference scenario with filler turns."""
    filler_turns = [
        "Explain what a neural network is.",
        "Give me a fun fact about space.",
        "What is the capital of France?",
        "How do I boil an egg?",
        "Tell me a joke.",
        "Suggest a movie.",
        "Define machine learning.",
        "How do I stay productive?",
    ]
    return (
        ["I like concise answers."]
        + filler_turns
        + [
            "Remind me what I said about answer style.",
            "Give me a short summary of our chat.",
        ]
    )


def _evaluate(preference_reply: str, summary_reply: str) -> Dict[str, bool]:
    """Evaluate preference recall and summary mention heuristics."""
    preference_recall_correct = _mentions_concise(preference_reply)
    summary_mentions_preference = _mentions_concise(summary_reply)
    return {
        "preference_recall_correct": preference_recall_correct,
        "summary_mentions_preference": summary_mentions_preference,
    }


def _mentions_concise(text: str) -> bool:
    """Check if text mentions concise preference."""
    lowered = text.lower()
    return "concise" in lowered or "short" in lowered


def main() -> None:
    """Entry point for running a demo experiment."""
    parser = argparse.ArgumentParser(description="Run LLM memory experiments.")
    parser.add_argument(
        "--mode",
        choices=["no_memory", "context", "retrieval", "hierarchical"],
        default="hierarchical",
        help="Memory mode to use for the experiment.",
    )
    parser.add_argument(
        "--scenario",
        choices=["short_preference", "long_preference"],
        default="short_preference",
        help="Scenario to run for evaluation.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="Number of repeats for evaluation.",
    )
    args = parser.parse_args()
    mode = args.mode
    if args.scenario == "long_preference":
        user_queries = _scenario_long_preference()
    else:
        user_queries = _scenario_short_preference()

    results: List[Dict[str, bool]] = []
    for repeat_index in range(args.repeat):
        print(f"Running experiment {repeat_index + 1}/{args.repeat} in mode: {mode}")
        assistant_responses = run_experiment(mode, user_queries)
        # Evaluate last two turns only; assume fixed scenario ordering.
        preference_reply = assistant_responses[-2] if len(assistant_responses) >= 2 else ""
        summary_reply = assistant_responses[-1] if len(assistant_responses) >= 1 else ""
        results.append(_evaluate(preference_reply, summary_reply))

    accuracy = {
        "preference_recall_correct": sum(
            1 for item in results if item["preference_recall_correct"]
        )
        / max(1, len(results)),
        "summary_mentions_preference": sum(
            1 for item in results if item["summary_mentions_preference"]
        )
        / max(1, len(results)),
    }
    summary = {
        "mode": mode,
        "scenario": args.scenario,
        "repeat": args.repeat,
        "accuracy": accuracy,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
