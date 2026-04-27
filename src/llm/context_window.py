"""Utilities for keeping long document context inside model limits."""

from __future__ import annotations

from typing import Any, Iterable


def clamp_text(text: str, max_chars: int) -> str:
    """Keep the beginning and end of long text, preserving local and concluding context."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text

    marker = "\n...[truncated for context budget]...\n"
    keep = max(0, max_chars - len(marker))
    head = keep // 2
    tail = keep - head
    return f"{text[:head]}{marker}{text[-tail:]}"


def compact_jsonish(items: Iterable[Any], max_chars: int) -> str:
    """Render simple values while staying under a character budget."""
    rendered = []
    used = 0
    for item in items:
        value = str(item)
        next_used = used + len(value) + (2 if rendered else 0)
        if max_chars > 0 and next_used > max_chars:
            rendered.append("...")
            break
        rendered.append(value)
        used = next_used
    return ", ".join(rendered)


def split_batches(items: list[str], batch_size: int) -> list[list[str]]:
    """Split items into non-empty fixed-size batches."""
    if batch_size <= 0:
        return [items]
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def build_section_context(section: dict, content_budget: int) -> str:
    """Format a section with hierarchical cues for long-context extraction."""
    path = " > ".join(section.get("path", []))
    siblings = compact_jsonish(section.get("sibling_titles", []), 600) or "None"
    child_titles = compact_jsonish(section.get("child_titles", []), 600) or "None"
    parent_summary = clamp_text(section.get("parent_spatial_summary", ""), 1800) or "None"
    own_summary = clamp_text(section.get("spatial_summary", ""), 1800) or "None"
    content = clamp_text(section.get("content", ""), content_budget)

    return (
        f"ID: {section['id']}\n"
        f"Path: {path or section.get('title', 'Untitled')}\n"
        f"Title: {section.get('title', 'Untitled')}\n"
        f"Sibling Titles: {siblings}\n"
        f"Child Titles: {child_titles}\n"
        f"Parent Spatial Summary: {parent_summary}\n"
        f"Current Spatial Summary: {own_summary}\n"
        f"Content:\n{content}"
    )
