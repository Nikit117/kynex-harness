"""Conversation intent router: classifies user input before expensive execution.

Categories:
  CHAT             - casual/general conversation, no documents needed
  SYNTHESIS        - transform existing session state ("show as table", "sort by")
  DOCUMENT_LOOKUP  - quick lookup in loaded documents
  DOCUMENT_RETRIEVAL - deeper document search/read
  STRUCTURED_EXTRACTION - full extraction task
  COMPLEX_INVESTIGATION - multi-document analysis, comparison
  AMBIGUOUS        - unclear, needs clarification
  COMMAND          - slash command (handled before this router)
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, List, Optional


class Intent(str, Enum):
    CHAT = "CHAT"
    SYNTHESIS = "SYNTHESIS"
    DOCUMENT_LOOKUP = "DOCUMENT_LOOKUP"
    DOCUMENT_RETRIEVAL = "DOCUMENT_RETRIEVAL"
    STRUCTURED_EXTRACTION = "STRUCTURED_EXTRACTION"
    COMPLEX_INVESTIGATION = "COMPLEX_INVESTIGATION"
    AMBIGUOUS = "AMBIGUOUS"
    COMMAND = "COMMAND"


_CHAT_WORDS = frozenset({
    "hi", "hey", "hello", "sup", "yo", "thanks", "thank", "bye",
    "good", "cool", "ok", "okay", "sure", "nice", "great", "awesome",
    "lol", "haha", "hmm", "bruh", "bro", "dude", "man",
    "how", "whats", "what's", "tell",
})

_SYNTHESIS_PATTERNS = [
    re.compile(r"^(show|give|display|render|format|present)\s+(that|it|this|them)\b", re.I),
    re.compile(r"^(as a|in a|into a)\s+(table|list|csv|json|markdown|bullet)", re.I),
    re.compile(r"^(sort|filter|group|only)\s+(by|show|the|it)", re.I),
    re.compile(r"^(summarize|recap|brief)\s+(that|it|this|the)", re.I),
    re.compile(r"\b(as a table|as a list|as json|as csv|as markdown)\s*$", re.I),
    re.compile(r"^only\s+(show|include|display)\b", re.I),
    re.compile(r"^(export|save)\s+(that|it|this)\b", re.I),
]

_EXTRACTION_PATTERNS = [
    re.compile(r"\b(extract|populate|fill|complete)\s+(all|every|the|each)\b", re.I),
    re.compile(r"\b(extract)\s+\w+\s+(from|by|per|for)\b", re.I),
    re.compile(r"\b(I\/O counts?|io counts?|record.?s? (by|per|for))\b", re.I),
]

_INVESTIGATION_PATTERNS = [
    re.compile(r"\b(compare|contrast|cross-reference|reconcile|contradict)\b", re.I),
    re.compile(r"\b(identify.+?(contradictions?|discrepancies?|inconsistencies?))\b", re.I),
    re.compile(r"\b(analyze|assess|evaluate).+?(architecture|design|approach)\b", re.I),
]

_FOLLOW_UP_PATTERNS = [
    re.compile(r"^(redundant|why|where|when|how|which|what|source|page)\??\s*$", re.I),
    re.compile(r"^(show me|tell me more|elaborate|explain)\b", re.I),
    re.compile(r"^(and|also|what about|how about)\b", re.I),
]


class ConversationRouter:
    """Classifies user prompts into intent categories.

    Cheap heuristic first; LLM classification only for truly ambiguous cases.
    """

    def __init__(self, has_documents: bool = False, has_session_state: bool = False):
        self.has_documents = has_documents
        self.has_session_state = has_session_state

    def classify(self, prompt: str, last_intent: Intent | None = None) -> Intent:
        stripped = prompt.strip()

        if stripped.startswith("/"):
            return Intent.COMMAND

        words = stripped.lower().split()
        if not words:
            return Intent.CHAT

        # Short greetings/casual
        if len(words) <= 4 and words[0] in _CHAT_WORDS:
            if not any(w in stripped.lower() for w in ("document", "extract", "find", "search")):
                return Intent.CHAT

        # Synthesis: transform existing state
        if self.has_session_state:
            for pat in _SYNTHESIS_PATTERNS:
                if pat.search(stripped):
                    return Intent.SYNTHESIS

        # Follow-up to previous answer
        if last_intent and last_intent not in (Intent.CHAT, Intent.COMMAND):
            for pat in _FOLLOW_UP_PATTERNS:
                if pat.search(stripped):
                    if last_intent == Intent.SYNTHESIS:
                        return Intent.SYNTHESIS
                    return Intent.DOCUMENT_RETRIEVAL

        # Full extraction request
        for pat in _EXTRACTION_PATTERNS:
            if pat.search(stripped):
                return Intent.STRUCTURED_EXTRACTION

        # Complex investigation
        for pat in _INVESTIGATION_PATTERNS:
            if pat.search(stripped):
                return Intent.COMPLEX_INVESTIGATION

        # General knowledge questions without documents
        if not self.has_documents:
            _doc_words = {"document", "file", "pdf", "page", "section", "table", "extract"}
            if not any(w in stripped.lower() for w in _doc_words):
                return Intent.CHAT

        # Document questions (has docs loaded)
        if self.has_documents:
            question_words = {"what", "where", "which", "how", "who", "when", "does", "is", "are"}
            if words[0] in question_words:
                return Intent.DOCUMENT_RETRIEVAL
            if "?" in stripped:
                return Intent.DOCUMENT_RETRIEVAL

        # General knowledge question (explain X, what is Y)
        explain_re = re.compile(r"^(explain|define|what is|what are|describe|tell me about)\b", re.I)
        if explain_re.search(stripped) and not self.has_documents:
            return Intent.CHAT

        # Ambiguous: has documents but not clearly document-related
        if self.has_documents:
            return Intent.DOCUMENT_LOOKUP

        return Intent.CHAT

    def update_state(
        self, has_documents: bool | None = None, has_session_state: bool | None = None
    ) -> None:
        if has_documents is not None:
            self.has_documents = has_documents
        if has_session_state is not None:
            self.has_session_state = has_session_state


class IntentCompiler:
    """Compiles natural language into structured intent for document queries."""

    def compile(self, prompt: str, intent: Intent) -> Dict[str, Any]:
        return {
            "raw_prompt": prompt,
            "intent": intent.value,
            "needs_extraction": intent in (
                Intent.STRUCTURED_EXTRACTION,
                Intent.COMPLEX_INVESTIGATION,
            ),
            "needs_search": intent in (
                Intent.DOCUMENT_LOOKUP,
                Intent.DOCUMENT_RETRIEVAL,
                Intent.STRUCTURED_EXTRACTION,
                Intent.COMPLEX_INVESTIGATION,
            ),
            "needs_synthesis": intent == Intent.SYNTHESIS,
            "needs_llm": intent not in (Intent.COMMAND,),
        }
