from src.metric.gaia_scorer import question_scorer
from src.metric.arc_scorer import arc_question_scorer


def get_scorer(dataset_type: str = "gaia"):
    """Return the appropriate scorer function for a given dataset type."""
    if dataset_type == "arc":
        return arc_question_scorer
    return question_scorer


__all__ = [
    "question_scorer",
    "arc_question_scorer",
    "get_scorer",
]