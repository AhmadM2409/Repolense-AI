"""Markdown sections, preserving preambles, fenced code and original source offsets."""

import re

from .base import BaseChunker, Chunk
from .semantic_chunker import SemanticChunker


class StructuralChunker(BaseChunker):
    SECTION_TOKEN_LIMIT = 800

    def __init__(self):
        self.semantic_chunker = SemanticChunker()
        self.encoder = self.semantic_chunker.encoder

    def chunk(self, text: str, metadata: dict) -> list[Chunk]:
        if not text or not text.strip():
            return []
        chunks = []
        for section in self._extract_sections(text):
            info = {**metadata, "heading_path": " > ".join(section["path"])}
            if section["level"]:
                info["heading_level"] = section["level"]
            if (
                len(self.encoder.encode(section["content"], disallowed_special=()))
                > self.SECTION_TOKEN_LIMIT
            ):
                pieces = self.semantic_chunker.chunk(section["content"], info)
                for piece in pieces:
                    piece.metadata["char_start"] += section["start"]
                    piece.metadata["char_end"] += section["start"]
                    chunks.append(piece)
            else:
                chunks.append(
                    Chunk(
                        section["content"],
                        {**info, "char_start": section["start"], "char_end": section["end"]},
                    )
                )
        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = index
        return chunks

    def _extract_sections(self, text: str) -> list[dict]:
        sections, stack = [], []
        start = offset = 0
        fence = None

        def append(end):
            if text[start:end].strip():
                sections.append(
                    {
                        "content": text[start:end],
                        "path": [item[1] for item in stack],
                        "level": stack[-1][0] if stack else 0,
                        "start": start,
                        "end": end,
                    }
                )

        for line in text.splitlines(keepends=True):
            marker = re.match(r"^[ ]{0,3}(`{3,}|~{3,})", line)
            if marker:
                char = marker.group(1)[0]
                if fence is None:
                    fence = (char, len(marker.group(1)))
                elif fence[0] == char and len(marker.group(1)) >= fence[1]:
                    fence = None
            elif fence is None:
                heading = re.match(r"^[ ]{0,3}(#{1,6})[ \t]+(.+)", line)
                if heading:
                    append(offset)
                    start = offset
                    level = len(heading.group(1))
                    while stack and stack[-1][0] >= level:
                        stack.pop()
                    stack.append((level, heading.group(2).strip()))
            offset += len(line)
        append(len(text))
        return sections
