# SMS Spam MLFlow

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Clasificacion de SMS spam con experimentos en MLflow. El notebook solo explora los datos. El entrenamiento esta en `src/train.py`.

## Como ejecutar

Desde esta carpeta, con el entorno del proyecto:

```text
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m src.train
```

Sin argumentos se usa el baseline: `C=1.0`, unigramas, `class_weight=balanced` y semilla `42`. La misma semilla fija la particion estratificada 80/20 y la regresion logistica. Dos ejecuciones con los mismos argumentos dan el mismo resultado.

El servidor de tracking tiene que estar activo:

```text
mlflow ui --host 127.0.0.1 --port 5000
```

El experimento se llama `TC5061 SMS Spam Hiperparametros`. La metrica principal es `f1_spam`.

Corridas de esta entrega, una por linea:

```text
.venv\Scripts\python -m src.train
.venv\Scripts\python -m src.train --C 0.1
.venv\Scripts\python -m src.train --C 10
.venv\Scripts\python -m src.train --ngram-max 2
.venv\Scripts\python -m src.train --class-weight none
```

## Organizacion

```
├── data
│   ├── external
│   ├── interim
│   ├── processed
│   └── raw            <- sms+spam+collection.zip
├── notebooks          <- exploracion del corpus
├── reports
│   └── figures
└── src
    ├── config.py
    ├── load_data.py
    ├── preprocess_data.py
    ├── train.py
    └── evaluate.py
```

`load_data.py` lee el corpus. `preprocess_data.py` limpia y guarda la tabla de modelado. `train.py` ajusta el pipeline y registra la corrida. `evaluate.py` calcula las metricas y la matriz de confusion.
