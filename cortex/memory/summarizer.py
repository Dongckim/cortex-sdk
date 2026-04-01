"""Rule-based context summarizer with hierarchical compression."""

import logging
from collections import Counter

from cortex.memory.context_store import ContextEvent

logger = logging.getLogger(__name__)


class Summarizer:
    """Hierarchical summarizer for context events.

    Compresses events in three tiers (paper Section III-D):
    - Recent (< 30s): full description preserved (~30 tokens)
    - Mid-range (30s-2min): compressed to key phrase (~80 tokens)
    - Old (> 2min): aggregated into group summary (~20 tokens)
    """

    def summarize_event(self, event: ContextEvent) -> str:
        """Generate a compressed summary for a single event.

        Extracts the first sentence or key phrase from the description.

        Args:
            event: Context event to summarize.

        Returns:
            Compressed summary string.
        """
        desc = event.description.strip()
        if not desc:
            return event.event_type

        # Take first sentence
        for sep in (".", "!", "?"):
            idx = desc.find(sep)
            if 0 < idx < 80:
                return desc[: idx + 1]

        # Truncate long descriptions
        if len(desc) > 60:
            return desc[:57] + "..."

        return desc

    def summarize_group(self, events: list[ContextEvent]) -> str:
        """Summarize a group of events into a single string.

        Args:
            events: List of events to summarize.

        Returns:
            Group summary string.
        """
        if not events:
            return ""

        type_counts = Counter(e.event_type for e in events)
        parts = [f"{count} {etype}" for etype, count in type_counts.items()]
        return f"[{len(events)} events: {', '.join(parts)}]"

    def hierarchical_summary(
        self, events: list[ContextEvent]
    ) -> dict[str, list[str]]:
        """Generate a three-tier hierarchical summary.

        Args:
            events: All active events (oldest first).

        Returns:
            Dict with 'recent', 'mid', 'old' tiers,
            each containing a list of summary strings.
        """
        recent: list[str] = []
        mid: list[str] = []
        old_events: list[ContextEvent] = []

        for event in events:
            age = event.age_seconds
            if age < 30:
                recent.append(f"[{event.age_label}] {event.description}")
            elif age < 120:
                mid.append(
                    f"[{event.age_label}] {self.summarize_event(event)}"
                )
            else:
                old_events.append(event)

        old: list[str] = []
        if old_events:
            old.append(self.summarize_group(old_events))

        logger.debug(
            "hierarchical: recent=%d mid=%d old=%d",
            len(recent), len(mid), len(old),
        )

        return {"recent": recent, "mid": mid, "old": old}

    def estimate_tokens(self, summary: dict[str, list[str]]) -> int:
        """Estimate token count for a hierarchical summary.

        Args:
            summary: Output from hierarchical_summary().

        Returns:
            Estimated token count.
        """
        total_chars = sum(
            len(s) for tier in summary.values() for s in tier
        )
        # Rough estimate: ~4 chars per token
        return max(1, total_chars // 4)
