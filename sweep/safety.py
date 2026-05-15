"""Lexicon danger: integer hit count (0–n) for targeted patterns and words."""
from __future__ import annotations

import re
from typing import Any

# Phrases — each successful pattern match counts as one hit (occurrences via findall).
_PHRASES_THREAT_SELF = (
    r"\bkill\s+yourself\b",
    r"\bkys\b",
    r"\bcommit\s+suicide\b",
    r"\bend\s+it\s+all\b",
    r"\bself[- ]harm\b",
    r"\bcut\s+myself\b",
    r"\bslit\s+(my|your)\s+wrist",
)
_PHRASES_THREAT_OTHER = (
    r"\bkill\s+(you|him|her|them|everyone)\b",
    r"\b(murder|assassinate|execute)\s+(you|him|her|them)",
    r"\b(go\s+)?die\b",
    r"\b(i('ll| will)\s+)?(stab|shoot|strangle)\s+you\b",
    r"\bterrorist\b",
    r"\bbomb(ing)?\b",
    r"\b(genocide|ethnic\s+cleansing)\b",
    r"\bmass\s+shooting\b",
)
_PHRASES_GORE = (
    r"\b(dismember|decapitate|behead)\w*\b",
    r"\b(gushing|splatter(ed)?)\s+blood\b",
    r"\btorture(s|d)?\b",
    r"\bmutilat\w*\b",
)
_PHRASES_SEXUAL = (
    r"\bchild\s+porn",
    r"\b(cp|csam)\b",
    r"\b(underage|minor)\s+sex",
    r"\brape\b",
    r"\bnon[- ]?consensual\b",
)
_PHRASES_HATE = (
    r"\b(ethnic|racial)\s+cleansing\b",
    r"\bholocaust\s+denial\b",
    r"\b gas\s+the\s+",
)

_WORDS_THREAT = frozenset({
    "murder", "kill", "killing", "killer", "slaughter", "massacre", "lynch",
    "stab", "stabbing", "strangle", "suicide", "genocide", "terrorism",
    "hostage", "kidnap", "abduct", "torture", "execute", "execution",
})
_WORDS_GORE = frozenset({
    "blood", "bloody", "gore", "gory", "carnage", "entrails", "dismember",
    "decapitate", "mutilate", "corpse", "cadaver",
})
_WORDS_PROFANITY = frozenset({
    "fuck", "fucking", "fucked", "shit", "bullshit", "asshole", "bitch",
    "bastard", "cunt", "dick", "piss", "damn", "hell",
})
_WORDS_DRUGS_ILLEGAL = frozenset({
    "cocaine", "heroin", "meth", "fentanyl", "cartel", "hitman", "mercenary",
})

_BAD_WORDS = _WORDS_THREAT | _WORDS_GORE | _WORDS_PROFANITY | _WORDS_DRUGS_ILLEGAL

_COMPILED_PHRASES: list[tuple[str, re.Pattern[str]]] = []
for label, patterns in (
    ("threat_self", _PHRASES_THREAT_SELF),
    ("threat_other", _PHRASES_THREAT_OTHER),
    ("gore", _PHRASES_GORE),
    ("sexual_illegal", _PHRASES_SEXUAL),
    ("hate", _PHRASES_HATE),
):
    for p in patterns:
        _COMPILED_PHRASES.append((label, re.compile(p, re.IGNORECASE)))

_WORD_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)


def danger_scan(text: str) -> tuple[int, dict[str, Any]]:
    """
    Count lexicon hits (non-negative integer).

    Each phrase pattern contributes len(findall) hits. Each token in the word lists
    contributes one hit per occurrence in the token stream.
    """
    if not text or not text.strip():
        return 0, {"hits": {}, "hit_total": 0}

    lowered = text.lower()
    hits: dict[str, int] = {}
    total = 0

    for label, rx in _COMPILED_PHRASES:
        n = len(rx.findall(lowered))
        if n:
            hits[label] = hits.get(label, 0) + n
            total += n

    for tok in _WORD_RE.findall(lowered):
        w = tok.lower()
        if w in _BAD_WORDS:
            key = f"word:{w}"
            hits[key] = hits.get(key, 0) + 1
            total += 1

    return total, {"hits": hits, "hit_total": total}
