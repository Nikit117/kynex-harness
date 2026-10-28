"""Structured conversational state for follow-up context and entity resolution.

Maintains entities, results, and evidence across turns so follow-up questions
like "redundant?", "what about ESD?", "compare both", "show as table" work
without reinvestigating documents.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class ConversationTurn:
    prompt: str
    answer: str
    intent: str = ""
    entities: List[str] = field(default_factory=list)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    structured_result: Optional[Dict[str, Any]] = None
    evidence_ids: List[int] = field(default_factory=list)
    reinvestigated: bool = True


class ConversationState:
    """Tracks structured conversational context across turns."""

    def __init__(self) -> None:
        self.turns: List[ConversationTurn] = []
        self.active_entities: List[str] = []
        self.active_documents: List[str] = []
        self.last_structured_result: Optional[Dict[str, Any]] = None
        self.comparison_entities: List[str] = []
        self.user_scope: str = ""

    @property
    def last_turn(self) -> Optional[ConversationTurn]:
        return self.turns[-1] if self.turns else None

    @property
    def last_answer(self) -> str:
        return self.turns[-1].answer if self.turns else ""

    @property
    def last_question(self) -> str:
        return self.turns[-1].prompt if self.turns else ""

    @property
    def last_citations(self) -> List[Dict[str, Any]]:
        return self.turns[-1].citations if self.turns else []

    def record_turn(
        self,
        prompt: str,
        answer: str,
        intent: str = "",
        citations: List[Dict[str, Any]] | None = None,
        structured_result: Dict[str, Any] | None = None,
        reinvestigated: bool = True,
    ) -> ConversationTurn:
        entities = self._extract_entities(prompt, answer)
        turn = ConversationTurn(
            prompt=prompt,
            answer=answer,
            intent=intent,
            entities=entities,
            citations=citations or [],
            structured_result=structured_result,
            evidence_ids=[c.get("id", 0) for c in (citations or [])],
            reinvestigated=reinvestigated,
        )
        self.turns.append(turn)
        if entities:
            self.active_entities = entities
        if structured_result is not None:
            self.last_structured_result = structured_result
        docs = set()
        for c in (citations or []):
            d = c.get("document", "")
            if d:
                docs.add(d)
        if docs:
            self.active_documents = sorted(docs)
        return turn

    def resolve_referent(self, prompt: str) -> str:
        """Resolve conversational references to concrete entities/context.

        Returns an enriched prompt with resolved entities prepended when
        the original is ambiguous, or the original prompt unchanged.
        """
        lower = prompt.strip().lower().rstrip("?").strip()

        if not self.turns:
            return prompt

        last = self.turns[-1]

        # Direct referents: "it", "that", "those", "both"
        if lower in ("it", "that", "this"):
            if self.active_entities:
                return f"Regarding {self.active_entities[0]}: {prompt}"
        if lower in ("both", "those", "them"):
            if len(self.active_entities) >= 2:
                ents = " and ".join(self.active_entities[:2])
                return f"Regarding {ents}: {prompt}"

        # "the first one", "the second one"
        ordinal_match = re.match(
            r"^the\s+(first|second|third)\s+one\b", lower
        )
        if ordinal_match:
            idx = {"first": 0, "second": 1, "third": 2}[ordinal_match.group(1)]
            if idx < len(self.active_entities):
                return f"Regarding {self.active_entities[idx]}: {prompt}"

        # "what about X?" — adds X as new entity while keeping context
        what_about = re.match(r"^what\s+about\s+(.+)$", lower)
        if what_about:
            new_entity = what_about.group(1).strip("? ")
            self.active_entities.append(new_entity)
            if len(self.active_entities) >= 2:
                self.comparison_entities = self.active_entities[-2:]
            return prompt

        # Single-word follow-ups that relate to the last entity
        if len(lower.split()) <= 2 and self.active_entities:
            adjective_words = {
                "redundant", "backup", "type", "model", "specification",
                "source", "page", "where", "how", "why",
            }
            if lower.split()[0] in adjective_words:
                entity = self.active_entities[0]
                return f"Is {entity} {prompt}?" if "?" not in prompt else f"Regarding {entity}: {prompt}"

        # "compare both", "compare them", "compare DCS and ESD"
        compare_match = re.match(r"^compare\s+(.+)$", lower)
        if compare_match:
            target = compare_match.group(1).strip()
            if target in ("both", "them", "those"):
                if self.comparison_entities:
                    ents = " and ".join(self.comparison_entities)
                    return f"Compare {ents}"
                elif len(self.active_entities) >= 2:
                    ents = " and ".join(self.active_entities[:2])
                    return f"Compare {ents}"

        # "source", "source?" — resolve to last evidence
        if lower in ("source", "source?", "where exactly", "where exactly?"):
            if last.citations:
                c = last.citations[0]
                return f"What is the source for {self.active_entities[0] if self.active_entities else 'that'}?"

        return prompt

    def has_result_for_synthesis(self) -> bool:
        """Check if there's a structured result available for synthesis."""
        return self.last_structured_result is not None or bool(
            self.turns and self.turns[-1].answer
        )

    def get_synthesis_context(self) -> Dict[str, Any]:
        """Get the context needed for synthesis (formatting existing results)."""
        return {
            "last_answer": self.last_answer,
            "last_question": self.last_question,
            "structured_result": self.last_structured_result,
            "active_entities": list(self.active_entities),
            "citations": self.last_citations,
        }

    def build_context_for_llm(self, turns: int = 3) -> str:
        """Build conversation context string for LLM calls."""
        lines = []
        for turn in self.turns[-turns:]:
            lines.append(f"User: {turn.prompt[:300]}")
            lines.append(f"KYNEX: {turn.answer[:500]}")
            if turn.entities:
                lines.append(f"[entities: {', '.join(turn.entities)}]")
        return "\n".join(lines)

    def _extract_entities(self, prompt: str, answer: str) -> List[str]:
        """Extract domain entities from prompt and answer."""
        entities = []
        combined = f"{prompt} {answer}"
        patterns = [
            r'\b(DCS|ESD|SIS|F&G|PLC|SCADA|HMI|MCC|VFD|UPS)\b',
            r'\b(AC\s*800M|S800)\b',
            r'\b(Area\s+\d+|Unit\s+\d+|Zone\s+\d+)\b',
            r'\b(I/O|IO)\s+(count|schedule|list|specification)\b',
            r'\b(cabinet|controller|processor|module|rack)\b',
        ]
        for pat in patterns:
            for m in re.finditer(pat, combined, re.IGNORECASE):
                entity = m.group(0).strip()
                if entity and entity not in entities:
                    entities.append(entity)
        return entities[:5]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for session persistence."""
        return {
            "active_entities": self.active_entities,
            "active_documents": self.active_documents,
            "comparison_entities": self.comparison_entities,
            "user_scope": self.user_scope,
            "turns": [
                {
                    "prompt": t.prompt[:2000],
                    "answer": t.answer[:20000],
                    "intent": t.intent,
                    "entities": t.entities,
                    "reinvestigated": t.reinvestigated,
                }
                for t in self.turns[-10:]
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationState":
        state = cls()
        state.active_entities = data.get("active_entities", [])
        state.active_documents = data.get("active_documents", [])
        state.comparison_entities = data.get("comparison_entities", [])
        state.user_scope = data.get("user_scope", "")
        for t in data.get("turns", []):
            state.turns.append(ConversationTurn(
                prompt=t.get("prompt", ""),
                answer=t.get("answer", ""),
                intent=t.get("intent", ""),
                entities=t.get("entities", []),
                reinvestigated=t.get("reinvestigated", True),
            ))
        return state
