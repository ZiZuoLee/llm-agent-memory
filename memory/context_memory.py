"""Context memory module for managing recent dialogue turns."""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List


class ContextMemory:
    """Maintain a sliding window of recent dialogue turns."""

    def __init__(self, max_turns: int) -> None:
        """Initialize memory with a maximum number of turns to retain."""
        if max_turns <= 0:
            raise ValueError("max_turns must be a positive integer.")
        self._max_turns = max_turns
        self._turns: Deque[Dict[str, str]] = deque(maxlen=max_turns)

    def add_turn(self, user_text: str, assistant_text: str) -> None:
        """Add a user/assistant turn to the memory."""
        self._turns.append({"user": user_text, "assistant": assistant_text})

    def get_context(self) -> List[Dict[str, str]]:
        """Return the stored turns in chronological order."""
        return list(self._turns)
