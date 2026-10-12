# TC5061 Experimentación con MLflow

Mendoza Mora José Luis

## Análisis

La mejor configuración es C=10, unigramas, class_weight=balanced y semilla 42. En la corrida auspicious-shark-466 el F1 de spam fue 0.9453, por encima del baseline con C=1.0 (F1 0.9049). Precisión y recall de spam quedaron en 0.9453. La exactitud fue 0.9864, frente al piso de 0.8760 de predecir siempre ham.

El hiperparámetro que más movió el F1 fue class_weight. Al quitar el balanceo, con el resto igual al baseline, el F1 bajó de 0.9049 a 0.8214 y el recall de spam de 0.9297 a 0.7188. El modelo deja de compensar que el spam es cerca del 12% de los mensajes: la precisión sube a 0.9583, pero se escapa mucho spam. C también importa. Con C=0.1 el F1 baja a 0.8864 y con C=10 sube a 0.9453. Incluir bigramas mejora el baseline a 0.9231, un cambio menor que relajar la regularización.

Las limitaciones son un solo split 80/20, sin validación cruzada, de modo que otra partición puede cambiar el orden. No se probó C=10 junto con bigramas. El modelo es una bolsa de palabras y no captura mensajes cuyo vocabulario no estaba en el entrenamiento. Cinco corridas no son una búsqueda. La semilla 42 hace repetible cada corrida, pero no estima la varianza entre particiones.

Comando de la mejor corrida:

```text
.venv\Scripts\python -m src.train --C 10
```

Repositorio:

https://github.com/zzJLMM/TC5061_Experimentacion_MLFlow_MendozaMoraJoseLuis

## Comparación de las cinco corridas

| Corrida | C | N-gramas | Peso | F1 spam |
| --- | --- | --- | --- | --- |
| auspicious-shark-466 | 10 | 1,1 | balanced | 0.9453 |
| masked-tern-717 | 1 | 1,2 | balanced | 0.9231 |
| shivering-shark-268 | 1 | 1,1 | balanced | 0.9049 |
| capricious-bass-750 | 0.1 | 1,1 | balanced | 0.8864 |
| unique-fly-720 | 1 | 1,1 | none | 0.8214 |

Semilla 42 en las cinco. El baseline es shivering-shark-268.

## Evidencias para pegar en Word

Formato de entrega: Arial o Times New Roman, tamaño 11, interlineado 1.5. Máximo dos cuartillas sin contar capturas. Nombre del archivo: TC5061_Experimentacion_MLFlow_MendozaMoraJoseLuis.pdf.

1. Captura de la tabla comparativa en MLflow. Experimento TC5061 SMS Spam Hiperparametros, vista Model training, Runs. Conviene mostrar las columnas f1_spam, C, ngram_range y class_weight. Dirección: http://127.0.0.1:5000/#/experiments/3/runs
2. Captura del detalle de la corrida auspicious-shark-466 con sus artefactos: parámetros, métricas, modelo sms_spam_model y la matriz de confusión. Dirección: http://127.0.0.1:5000/#/experiments/3/runs/197c1cb190fc491fbc489dc52c19c7db
3. Figura local de esa matriz: reports/figures/confusion_matrix_c10.0_ngram1_balanced.png

La corrida auspicious-shark-466 quedó Finished. El experimento local está en http://127.0.0.1:5000.
