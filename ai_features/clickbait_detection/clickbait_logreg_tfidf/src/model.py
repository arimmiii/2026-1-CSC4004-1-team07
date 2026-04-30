from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression


def build_model(
    random_state: int = 42,
    c: float = 10.0,
    word_max_features: int = 50000,
    char_max_features: int = 50000,
) -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(
                                analyzer="word",
                                ngram_range=(1, 2),
                                max_features=word_max_features,
                                min_df=2,
                                max_df=0.98,
                                sublinear_tf=True,
                            ),
                        ),
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char",
                                ngram_range=(2, 5),
                                max_features=char_max_features,
                                min_df=2,
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=c,
                    solver="saga",
                    max_iter=300,
                    random_state=random_state,
                    verbose=1,
                ),
            ),
        ]
    )
