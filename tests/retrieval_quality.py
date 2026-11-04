"""Binary gold relevance metrics; labels are independent of ranked output."""

import math


def metrics(paths, relevant):
    if not relevant:
        return None
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate evaluation result")
    gold = set(relevant)
    first = next((i for i, p in enumerate(paths[:10], 1) if p in gold), None)
    dcg = sum(1 / math.log2(i + 1) for i, p in enumerate(paths[:10], 1) if p in gold)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(10, len(gold)) + 1))
    return {
        "recall_at_5": len(set(paths[:5]) & gold) / len(gold),
        "recall_at_10": len(set(paths[:10]) & gold) / len(gold),
        "mrr_at_10": 1 / first if first else 0.0,
        "ndcg_at_10": dcg / ideal,
    }
