from __future__ import annotations

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC


def build_model(
    random_state: int = 42,
    svd_components: int = 256,
    rbf_components: int = 512,
    gamma: float = 0.7,
    c: float = 1.0,
) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                ),
            ),
            ("svd", TruncatedSVD(n_components=svd_components, random_state=random_state)),
            ("norm", Normalizer(copy=False)),
            ("rbf", RBFSampler(gamma=gamma, n_components=rbf_components, random_state=random_state)),
            ("clf", LinearSVC(C=c, class_weight="balanced", random_state=random_state)),
        ]
    )
