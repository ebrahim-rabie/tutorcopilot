from __future__ import annotations


class CrossEncoderReranker:
    """Cross-encoder scoring over (query, chunk) pairs. Downloads on first use."""

    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [float(s) for s in self.model.predict(pairs)]
