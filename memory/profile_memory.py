"""Profile memory module for storing user preferences and style."""

from __future__ import annotations

from typing import Dict


class ProfileMemory:
    """Structured key-value store for lightweight user profile signals."""

    def __init__(self) -> None:
        """Initialize profile with predefined keys."""
        self._profile: Dict[str, str] = {"preference": "", "style": ""}

    def update_from_text(self, text: str) -> None:
        """Update profile fields using simple rule-based heuristics."""
        lowered = text.lower()

        preference = self._extract_preference(lowered)
        if preference:
            self._profile["preference"] = preference

        style = self._extract_style(lowered)
        if style:
            self._profile["style"] = style

    def get_profile_prompt(self) -> str:
        """Format the stored profile into a prompt string."""
        lines = []
        if self._profile["preference"]:
            lines.append(f"User preference: {self._profile['preference']}.")
        if self._profile["style"]:
            lines.append(f"User style: {self._profile['style']}.")
        return "\n".join(lines)

    def _extract_preference(self, lowered: str) -> str:
        """Extract preference signals from text."""
        if "i don't like" in lowered or "i do not like" in lowered or "i dislike" in lowered:
            return "avoid what the user dislikes"
        if "i like" in lowered:
            return "follow what the user likes"
        if "i prefer" in lowered or "my preference is" in lowered:
            return "follow the user's stated preference"
        return ""

    def _extract_style(self, lowered: str) -> str:
        """Extract style signals from text."""
        if "formal" in lowered:
            return "formal"
        if "casual" in lowered or "informal" in lowered:
            return "casual"
        if "concise" in lowered or "brief" in lowered:
            return "concise"
        if "detailed" in lowered or "thorough" in lowered:
            return "detailed"
        return ""
