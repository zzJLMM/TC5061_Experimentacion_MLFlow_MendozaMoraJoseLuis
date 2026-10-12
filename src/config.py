"""Rutas del proyecto y valores por defecto del entrenamiento."""

from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

FIGURES_DIR = PROJ_ROOT / "reports" / "figures"

CORPUS_ZIP_NAME = "sms+spam+collection.zip"
CORPUS_FILE_NAME = "SMSSpamCollection"
LOADED_MESSAGES_NAME = "messages_loaded.csv"
PROCESSED_MESSAGES_NAME = "messages.csv"

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "TC5061 SMS Spam Hiperparametros"
MODEL_ARTIFACT_NAME = "sms_spam_model"

DEFAULT_REGULARIZATION = 1.0
DEFAULT_NGRAM_MAX = 1
DEFAULT_CLASS_WEIGHT = "balanced"
DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_SOLVER = "lbfgs"
DEFAULT_MAX_ITERATIONS = 1000

EXPECTED_CLEAN_ROWS = 5160
