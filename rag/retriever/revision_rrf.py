"""Equal-weight exact RRF over verified canonical branch matches; no shared state."""

import math
from fractions import Fraction

from api.schemas.search import HybridMatch

RRF_K = 60
CITATION = ("source_id", "chunk_id", "path", "commit_sha", "line_start", "line_end", "excerpt")


def fuse(keyword, semantic, top_k):
    """Deduplicate each branch by best one-based rank, rejecting inconsistent citations."""
    if type(top_k) is not int or not 1 <= top_k <= 20:
        raise ValueError("Invalid fusion limit")
    records = {}
    for branch, candidates in (("keyword", keyword), ("semantic", semantic)):
        if len(candidates) > 20:
            raise ValueError("Fusion candidate limit exceeded")
        for item in candidates:
            rank = item.rank
            value = item.bm25_score if branch == "keyword" else item.cosine_distance
            if (
                type(rank) is not int
                or not 1 <= rank <= 20
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or (branch == "keyword" and value <= 0)
                or (branch == "semantic" and not -1e-6 <= value <= 2.000001)
            ):
                raise ValueError("Invalid fusion branch metadata")
            citation = {name: getattr(item, name) for name in CITATION}
            key = str(item.chunk_id)
            row = records.setdefault(key, {"citation": citation, "branches": {}})
            if row["citation"] != citation:
                raise ValueError("Fusion provenance disagreement")
            previous = row["branches"].get(branch)
            if previous is not None and previous[0] == rank and previous[1] != value:
                raise ValueError("Fusion duplicate score disagreement")
            if previous is None or rank < previous[0]:
                row["branches"][branch] = (rank, value)
    ordered = []
    for key, row in records.items():
        branches = row["branches"]
        score = sum((Fraction(1, RRF_K + pair[0]) for pair in branches.values()), Fraction())
        ordered.append((score, min(pair[0] for pair in branches.values()), key, row))
    ordered.sort(key=lambda item: (-item[0], item[1], item[2]))
    output = []
    for position, (score, _, _, row) in enumerate(ordered[:top_k], 1):
        kw = row["branches"].get("keyword")
        sem = row["branches"].get("semantic")
        output.append(
            HybridMatch(
                **row["citation"],
                rank=position,
                rrf_score=float(score),
                keyword_rank=kw[0] if kw else None,
                semantic_rank=sem[0] if sem else None,
                bm25_score=kw[1] if kw else None,
                cosine_distance=sem[1] if sem else None,
                retrieval_source="both" if kw and sem else "keyword" if kw else "semantic",
            )
        )
    return output
