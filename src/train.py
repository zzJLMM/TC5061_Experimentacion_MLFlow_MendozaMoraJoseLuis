"""Entrena el clasificador de SMS y registra la corrida en MLflow."""

import argparse
import os
from pathlib import Path
import timeit

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import mlflow
from mlflow.models import infer_signature
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    DEFAULT_CLASS_WEIGHT,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_NGRAM_MAX,
    DEFAULT_REGULARIZATION,
    DEFAULT_SEED,
    DEFAULT_SOLVER,
    DEFAULT_TEST_SIZE,
    EXPERIMENT_NAME,
    FIGURES_DIR,
    MODEL_ARTIFACT_NAME,
    TRACKING_URI,
)
from src.evaluate import compute_metrics, save_confusion_matrix
from src.load_data import load_messages, save_loaded_messages
from src.preprocess_data import clean_messages, save_messages


def parse_arguments() -> argparse.Namespace:
    """Expone hiperparametros y semilla. Sin argumentos usa el baseline."""
    parser = argparse.ArgumentParser(
        description=(
            "Entrena un clasificador de SMS spam y registra la corrida."
        )
    )
    parser.add_argument(
        "--C",
        type=float,
        default=DEFAULT_REGULARIZATION,
        help="Inversa de la regularizacion L2. Default: 1.0.",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=DEFAULT_NGRAM_MAX,
        help="Tope de n-gramas. 1 es unigrama y 2 incluye bigramas.",
    )
    parser.add_argument(
        "--class-weight",
        choices=["balanced", "none"],
        default=DEFAULT_CLASS_WEIGHT,
        help="Peso de las clases. balanced compensa el spam minoritario.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Semilla de la particion y del clasificador. Default: 42.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=TRACKING_URI,
        help="Servidor de MLflow. Default: http://127.0.0.1:5000.",
    )
    return parser.parse_args()


def build_text_pipeline(
    regularization: float,
    ngram_max: int,
    class_weight: str,
    seed: int,
) -> Pipeline:
    """Arma TF-IDF y regresion logistica. El vectorizador ve solo train."""
    weight = None if class_weight == "none" else class_weight
    vectorizer = TfidfVectorizer(ngram_range=(1, ngram_max))
    classifier = LogisticRegression(
        C=regularization,
        solver=DEFAULT_SOLVER,
        max_iter=DEFAULT_MAX_ITERATIONS,
        random_state=seed,
        class_weight=weight,
    )
    text_features = ColumnTransformer(
        [("message_text", vectorizer, "message")]
    )
    return Pipeline(
        steps=[
            ("term_frequency", text_features),
            ("classifier", classifier),
        ]
    )


def split_messages(
    messages: pd.DataFrame,
    labels: pd.Series,
    test_size: float,
    seed: int,
):
    """Parte de forma estratificada para conservar la proporcion de spam."""
    return train_test_split(
        messages,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )


def collect_parameters(arguments: argparse.Namespace) -> dict:
    """Reune lo que se registra en la corrida, incluidos los valores fijos."""
    return {
        "C": arguments.C,
        "ngram_range": f"1,{arguments.ngram_max}",
        "class_weight": arguments.class_weight,
        "seed": arguments.seed,
        "solver": DEFAULT_SOLVER,
        "max_iter": DEFAULT_MAX_ITERATIONS,
        "test_size": DEFAULT_TEST_SIZE,
        "vectorizer": "TfidfVectorizer",
    }


def fit_pipeline(
    pipeline: Pipeline,
    messages: pd.DataFrame,
    labels: pd.Series,
):
    """Ajusta una sola vez y devuelve los segundos que tardo el fit."""
    timer = timeit.Timer(lambda: pipeline.fit(messages, labels))
    return timer.timeit(number=1)


def log_training_run(
    pipeline: Pipeline,
    parameters: dict,
    metrics: dict,
    training_seconds: float,
    figure_path: Path,
    train_messages: pd.DataFrame,
    tracking_uri: str,
) -> str:
    """Registra parametros, metricas, la matriz y el pipeline."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    example = train_messages.iloc[:5]
    signature = infer_signature(example, pipeline.predict(example))

    with mlflow.start_run() as run:
        mlflow.log_params(parameters)
        mlflow.log_metrics(metrics)
        mlflow.log_metric("training_seconds", float(training_seconds))
        mlflow.log_artifact(str(figure_path))
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name=MODEL_ARTIFACT_NAME,
            signature=signature,
            input_example=example,
        )
        return run.info.run_id


def main() -> None:
    arguments = parse_arguments()
    loaded = load_messages()
    save_loaded_messages(loaded)
    messages = clean_messages(loaded)
    save_messages(messages)

    features = messages[["message"]]
    labels = (messages["label"] == "spam").astype(int)
    train_messages, test_messages, train_labels, test_labels = split_messages(
        features,
        labels,
        DEFAULT_TEST_SIZE,
        arguments.seed,
    )
    pipeline = build_text_pipeline(
        arguments.C,
        arguments.ngram_max,
        arguments.class_weight,
        arguments.seed,
    )
    training_seconds = fit_pipeline(pipeline, train_messages, train_labels)
    predictions = pipeline.predict(test_messages)
    metrics = compute_metrics(test_labels, predictions)
    figure_path = FIGURES_DIR / (
        f"confusion_matrix_c{arguments.C}"
        f"_ngram{arguments.ngram_max}"
        f"_{arguments.class_weight}.png"
    )
    save_confusion_matrix(test_labels, predictions, figure_path)
    run_id = log_training_run(
        pipeline,
        collect_parameters(arguments),
        metrics,
        training_seconds,
        figure_path,
        train_messages,
        arguments.tracking_uri,
    )

    print(f"run_id: {run_id}")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print(f"training_seconds: {training_seconds:.4f}")


if __name__ == "__main__":
    main()
