
"""
ShopSense Session Module
Manages session-based personalization.
Used by Phase 9 FastAPI backend.

Usage:
    from session import SessionMemory, update_session_after_search
    from session import record_like, record_dismiss
"""
from dataclasses import dataclass, field
from collections import Counter
import re

STOPWORDS = {'the', 'a', 'an', 'for', 'and', 'or', 'in', 'on',
             'at', 'to', 'with', 'of', 'is', 'it', 'this', 'that',
             'my', 'me', 'i', 'set', 'inch', 'pack', 'piece'}


def extract_keywords(text):
    words = re.findall(r'\b[a-z]{4,}\b', text.lower())
    return [w for w in words if w not in STOPWORDS]


@dataclass
class SessionMemory:
    liked_asins     : list    = field(default_factory=list)
    dismissed_asins : list    = field(default_factory=list)
    search_history  : list    = field(default_factory=list)
    seen_asins      : set     = field(default_factory=set)
    price_points    : list    = field(default_factory=list)
    keyword_counts  : Counter = field(default_factory=Counter)
    liked_categories: list    = field(default_factory=list)
    liked_keywords  : list    = field(default_factory=list)
    avg_price_liked : float   = field(default=None)
    turn_count      : int     = field(default=0)

    def to_context_string(self):
        if self.turn_count == 0:
            return 'No session history yet.'
        parts = []
        if self.search_history:
            parts.append(f'Recent searches: {', '.join(repr(q) for q in self.search_history[-3:])}' )
        if self.liked_asins:
            parts.append(f'Liked {len(self.liked_asins)} product(s)')
        if self.liked_keywords:
            parts.append(f'Liked keywords: {', '.join(self.liked_keywords[:5])}')
        if self.avg_price_liked:
            parts.append(f'Avg liked price: ${self.avg_price_liked:.2f}')
        return ' | '.join(parts) if parts else 'Session started.'

    def reset(self):
        self.__init__()
