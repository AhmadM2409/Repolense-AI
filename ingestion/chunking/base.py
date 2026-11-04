from dataclasses import dataclass
import hashlib
import json

@dataclass
class Chunk:
    """Legacy-compatible text/metadata chunk, shared with vector-store adapters.

    Missing line references mean uncitable transformed text; never infer source lines.
    """
    text: str
    metadata: dict

    @property
    def source_id(self) -> str:
        value = self.metadata.get("source_id")
        if not value:
            raise ValueError("Chunk requires source_id before vector persistence")
        return str(value)

    @property
    def chunk_index(self) -> int:
        value = self.metadata.get("chunk_index", 0)
        if not isinstance(value, int) or value < 0:
            raise ValueError("chunk_index must be nonnegative")
        return value

    @property
    def id(self) -> str:
        return f"{self.source_id}_chunk_{self.chunk_index}"

    @property
    def section(self) -> str:
        return self.metadata.get("heading_path") or self.metadata.get("section") or ""

    def vector_metadata(self) -> dict:
        # Chroma supports scalar values only; retain nested data as deterministic JSON.
        result = {}
        for key, value in self.metadata.items():
            if value is None:
                continue
            result[key] = value if isinstance(value, (str, int, float, bool)) else json.dumps(value, sort_keys=True)
        result.update(source_id=self.source_id, chunk_index=self.chunk_index, section=self.section)
        return result

class BaseChunker:
    def chunk(self, text: str, metadata: dict) -> list[Chunk]:
        raise NotImplementedError("Subclasses must implement chunk()")

def source_line_chunks(text: str, metadata: dict, lines_per_chunk: int = 80) -> list[Chunk]:
    """Lossless source chunks with exact inclusive lines; no tokenizer/network requirement."""
    if lines_per_chunk < 1:
        raise ValueError("lines_per_chunk must be positive")
    lines = text.splitlines()
    return [Chunk("\n".join(lines[start:start+lines_per_chunk]),
                  {**metadata, "chunk_index": start//lines_per_chunk,
                   "line_start": start+1, "line_end": min(start+lines_per_chunk,len(lines))})
            for start in range(0,len(lines),lines_per_chunk) if "\n".join(lines[start:start+lines_per_chunk]).strip()]
