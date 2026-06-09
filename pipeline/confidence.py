"""Confidence blending and review-flag logic."""


def compute_confidence(
    llm_confidence: float,
    retrieval_score: float,
    has_enrichment: bool = False,
    threshold: float = 0.75,
) -> tuple[float, bool]:
    """Blend LLM and retrieval scores into a final confidence, then set the review flag.

    Weights:
      - LLM confidence carries 70 % of the signal (it saw the full prompt).
      - Retrieval cosine similarity carries 30 % (quality of the candidate set).
      - A small enrichment bonus (+0.03) is applied when the record had extra
        data (e.g. a matched chain name) that raised retrieval quality.

    The review flag is True whenever the blended score falls below *threshold*.

    Args:
        llm_confidence: Confidence reported by the LLM (0–1).
        retrieval_score: Cosine similarity of the top retrieval hit (0–1).
        has_enrichment: Whether the record benefited from enrichment signals.
        threshold: Records scoring below this trigger a manual-review flag.

    Returns:
        (final_confidence, needs_review) tuple.
    """
    blended = 0.70 * llm_confidence + 0.30 * retrieval_score
    if has_enrichment:
        blended = min(1.0, blended + 0.03)
    final = round(blended, 4)
    return final, final < threshold
