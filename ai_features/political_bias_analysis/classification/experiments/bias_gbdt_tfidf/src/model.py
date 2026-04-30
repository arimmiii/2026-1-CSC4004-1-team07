from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer


def build_model(
    model_type: str = "lightgbm",
    random_state: int = 42,
    svd_components: int = 220,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 8,
    num_leaves: int = 63,
    word_max_features: int = 120000,
    char_max_features: int = 180000,
    n_jobs: int = 2,
    use_char_features: bool = True,
    num_class: int = 5,
):
    features = [
        (
            "word",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.98,
                sublinear_tf=True,
                max_features=word_max_features,
                dtype=np.float32,
            ),
        ),
    ]

    if use_char_features and char_max_features > 0:
        features.append(
            (
                "char",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 5),
                    min_df=2,
                    sublinear_tf=True,
                    max_features=char_max_features,
                    dtype=np.float32,
                ),
            )
        )

    if model_type == "lightgbm":
        from lightgbm import LGBMClassifier

        clf = LGBMClassifier(
            objective="multiclass",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=-1,
        )
    elif model_type == "xgboost":
        from xgboost import XGBClassifier

        clf = XGBClassifier(
            objective="multi:softprob",
            num_class=num_class,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            random_state=random_state,
            n_jobs=n_jobs,
            eval_metric="mlogloss",
        )
    else:
        raise ValueError("model_type must be one of: lightgbm, xgboost")

    return Pipeline(
        [
            ("features", FeatureUnion(features)),
            ("svd", TruncatedSVD(n_components=svd_components, random_state=random_state)),
            ("norm", Normalizer(copy=False)),
            ("clf", clf),
        ]
    )
