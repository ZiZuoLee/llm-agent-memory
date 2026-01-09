"""Prompt builder module for constructing LLM chat messages."""

from __future__ import annotations

from typing import Dict, List


class PromptBuilder:
    """Build chat messages from memories and current user query."""

    def build_messages(
        self,
        context_memories: List[Dict[str, str]],
        retrieved_memories: List[str],
        profile_prompt: str,
        user_query: str,
    ) -> List[Dict[str, str]]:
        """Construct OpenAI chat messages with profile, memories, and query."""
        messages: List[Dict[str, str]] = []

        if profile_prompt:
            messages.append({"role": "system", "content": profile_prompt})

        if retrieved_memories:
            memories_text = "\n".join(f"- {item}" for item in retrieved_memories)
            messages.append(
                {
                    "role": "system",
                    "content": f"Relevant memories:\n{memories_text}",
                }
            )

        for turn in context_memories:
            user_text = turn.get("user", "")
            assistant_text = turn.get("assistant", "")
            if user_text:
                messages.append({"role": "user", "content": user_text})
            if assistant_text:
                messages.append({"role": "assistant", "content": assistant_text})

        messages.append({"role": "user", "content": user_query})
        return messages
