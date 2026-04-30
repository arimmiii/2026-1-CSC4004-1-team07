from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_model(random_state: int = 42, c: float = 4.0) -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                    max_features=80000,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=c,
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=200,
                    random_state=random_state,
                ),
            ),
        ]
    )
