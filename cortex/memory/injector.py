"""Context injector — assembles VLM prompts with compressed context."""

import logging

from cortex.memory.context_store import ContextStore
from cortex.memory.retriever import Retriever
from cortex.memory.summarizer import Summarizer

logger = logging.getLogger(__name__)


class ContextInjector:
    """Assembles final VLM prompts with compressed context.

    Combines hierarchical summary from the context store with
    query-relevant retrieval results into a compact context block.

    Args:
        store: Context event store.
        summarizer: Summarizer instance. Creates default if None.
        retriever: Retriever instance. Creates default if None.
    """

    def __init__(
        self,
        store: ContextStore,
        summarizer: Summarizer | None = None,
        retriever: Retriever | None = None,
    ) -> None:
        self._store = store
        self._summarizer = summarizer or Summarizer()
        self._retriever = retriever or Retriever()

    def build_context(self, query: str | None = None) -> str:
        """Build a compressed context string for VLM injection.

        Args:
            query: Optional query to retrieve relevant past events.

        Returns:
            Formatted context string ready for VLM prompt injection.
        """
        events = self._store.events
        if not events:
            return ""

        parts: list[str] = []

        # Hierarchical summary
        summary = self._summarizer.hierarchical_summary(events)

        if summary["old"]:
            parts.append("Background: " + " ".join(summary["old"]))

        if summary["mid"]:
            parts.append("Earlier: " + " | ".join(summary["mid"]))

        if summary["recent"]:
            parts.append("Recent: " + " | ".join(summary["recent"]))

        # Query-relevant retrieval
        if query:
            results = self._retriever.search(query, events, top_k=3)
            relevant = [
                (e, s) for e, s in results if s > 0.1
            ]
            if relevant:
                relevant_strs = [
                    f"[{e.age_label}] {e.description}" for e, _ in relevant
                ]
                parts.append("Relevant: " + " | ".join(relevant_strs))

        context = "\n".join(parts)

        tokens = self._summarizer.estimate_tokens(summary)
        logger.debug("context built: %d chars, ~%d tokens", len(context), tokens)

        return context

    def inject_prompt(
        self, user_prompt: str, system_context: str = ""
    ) -> str:
        """Build a full prompt with context injected.

        Args:
            user_prompt: The user's current question/request.
            system_context: Optional system-level context.

        Returns:
            Full prompt string with context block prepended.
        """
        context = self.build_context(query=user_prompt)

        parts: list[str] = []

        if system_context:
            parts.append(system_context)

        if context:
            parts.append(f"[Context]\n{context}\n[/Context]")

        parts.append(user_prompt)

        return "\n\n".join(parts)

    @property
    def context_stats(self) -> dict:
        """Current context statistics."""
        events = self._store.events
        summary = self._summarizer.hierarchical_summary(events)
        return {
            "total_events": len(events),
            "recent_count": len(summary["recent"]),
            "mid_count": len(summary["mid"]),
            "old_count": len(summary["old"]),
            "estimated_tokens": self._summarizer.estimate_tokens(summary),
        }
