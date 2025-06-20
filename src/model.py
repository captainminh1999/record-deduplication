"""Train a model to predict duplicate records."""

from sklearn.linear_model import LogisticRegression
import pandas as pd


def main(features_path: str = "data/features.csv", labels_path: str = "data/labels.csv"):
    """Train logistic regression on labeled pairs and score all pairs."""
    X = pd.read_csv(features_path, index_col=[0, 1])
    y = pd.read_csv(labels_path)["is_dup"]

    model = LogisticRegression(solver="liblinear", max_iter=500)
    model.fit(X.loc[y.index], y)

    X["prob"] = model.predict_proba(X)[:, 1]
    X[X["prob"] > 0.9].to_csv("data/dupes_high_conf.csv")


if __name__ == "__main__":
    main()
