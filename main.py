"""
Entry point for the baseline predictive pipeline.

Run with:
    python main.py

This orchestrates the full (deliberately simple) pipeline:
    load config -> load data -> preprocess -> split -> train
    -> evaluate (train & test) -> save results
"""
import yaml

from src.data import load_data
from src.model import build_model, build_pipeline
from src.evaluate import evaluate, fairness_report, oof_classification_report, cross_validate_pipeline, cv_report
from src.results import save_run
from src.preprocessing import (
    clean_dataset, split_features_target, split_dev_test,
)
from sklearn.model_selection import StratifiedKFold


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    df_raw = load_data(config["data"]["path"])
    df_clean = clean_dataset(df_raw, config["diagnostics"])

    '''
    X_train, X_test, y_train, y_test, extras_test = preprocess(
        df_clean,
        target=config["data"]["target"],
        sensitive_attr=config["data"]["sensitive_attr"],
        drop_columns=config["data"]["drop_columns"],
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
    )
    '''
    mnar_sources = config["preprocessing"].get("mnar_indicator_sources", [])
    X, y, extras = split_features_target(df_clean, config["data"], mnar_sources)
    
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = split_dev_test(
        X, y, extras,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
    )

    pipeline = build_pipeline(config["preprocessing"], config["model"])
    pipeline.fit(X_dev, y_dev)
    '''
    model = build_model(config["model"])
    model.fit(X_train, y_train)
    
    # predict on both splits -- train accuracy vs. test accuracy is how we'll spot overfitting, not just how "good" the model looks
    y_train_pred = pipeline.predict(X_dev)
    y_test_pred = pipeline.predict(X_test)
    report = evaluate(y_dev, y_train_pred, y_test, y_test_pred)
    '''
    cv_config = config["cv"]
    scoring = cv_config.get("scoring", "accuracy")
    shuffle = cv_config.get("shuffle", True)
    n_jobs = cv_config.get("n_jobs", 1)

    cv = StratifiedKFold(n_splits=cv_config["n_splits"], shuffle=shuffle,
                     random_state=cv_config.get("random_state") if shuffle else None)
    fold_scores, y_oof = cross_validate_pipeline(
    pipeline, X_dev, y_dev, cv, scoring, n_jobs=n_jobs
    )
    report =""+ cv_report(fold_scores, scoring)

    report += "\n\n" + oof_classification_report(y_dev, y_oof)
    report += "\n" + fairness_report(y_dev, y_oof, extras_dev, sensitive_attr=config["data"]["sensitive_attr"])

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()
