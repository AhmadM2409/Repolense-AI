"""Real BPE-bounded source slices with original character offsets."""

import re

import tiktoken

from .base import BaseChunker, Chunk


class SemanticChunker(BaseChunker):
    """Prefer sentence boundaries; split oversized sentences, retain <=50-token overlap."""

    TARGET_CHUNK_TOKENS = 500
    OVERLAP_TOKENS = 50
    MAX_TOKENS = 800

    def __init__(self):
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def _count(self, text):
        return len(self.encoder.encode(text, disallowed_special=()))

    def chunk(self, text: str, metadata: dict) -> list[Chunk]:
        if not text or not text.strip():
            return []
        boundaries = [match.end() for match in re.finditer(r"(?<=[.!?])\s+|\n{2,}", text)]
        chunks = []
        start = 0
        while start < len(text):
            low, high = start + 1, len(text)
            while low <= high:
                mid = (low + high) // 2
                if self._count(text[start:mid]) <= self.TARGET_CHUNK_TOKENS:
                    low = mid + 1
                else:
                    high = mid - 1
            end = max(start + 1, high)
            if end < len(text):
                candidates = [point for point in boundaries if start < point <= end]
                if candidates:
                    end = candidates[-1]
            value = text[start:end]
            if value.strip():
                chunks.append(
                    Chunk(
                        value,
                        {
                            **metadata,
                            "chunk_index": len(chunks),
                            "char_start": start,
                            "char_end": end,
                        },
                    )
                )
            if end == len(text):
                break
            if self._count(value) <= self.OVERLAP_TOKENS:
                start = end
                continue
            low, high = start + 1, end
            while low < high:
                mid = (low + high) // 2
                if self._count(text[mid:end]) <= self.OVERLAP_TOKENS:
                    high = mid
                else:
                    low = mid + 1
            # Short sentence slices must still advance; no infinite overlap loop.
            start = max(start + 1, low)
        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        pattern = r"(?<=[.!?])\s+(?=[A-Z])|(?<=\n)(?=[-•*])|(?<=\n)(?=[A-Z])"
        return [part.strip() for part in re.split(pattern, text) if part.strip()]
