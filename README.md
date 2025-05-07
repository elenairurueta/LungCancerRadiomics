# LungCancerRadiomics

Este proyecto proporciona herramientas para realizar análisis radiómicos en datos de cáncer de pulmón. A continuación, se incluyen las instrucciones para ejecutar los scripts.

## Búsqueda en GridSearch con `GS.py`

Se puede realizar una búsqueda en GridSearch para ajustar hiperparámetros utilizando el script `GS.py`. Este script acepta los siguientes argumentos:

- `--feature_selection`: El método de selección de características (por ejemplo, `anova`).
- `--model`: El modelo de aprendizaje automático a utilizar (por ejemplo, `random_forest`).
- `--hyperparameters`: Una cadena JSON que especifica las opciones de hiperparámetros.
- `--out_path`: El directorio de salida para los resultados.
- `--split_csv`: El archivo CSV que contiene la división de entrenamiento y prueba.

### Ejemplo

```bash
python GS.py --feature_selection anova --model random_forest --hyperparameters '{"n_trees": [50, 100], "max_depth": [1, 2, 3], "min_samples_split": [0.1, 0.5], "k": [5, 10, 20, 40]}' --out_path ".\data" --split_csv ".\data\train_5cv.csv"
```

## Análisis principal con `main.py`

El script `main.py` permite realizar el análisis principal. Este script acepta los siguientes argumentos:

- `--correlation_method`: El método de correlación a utilizar (por ejemplo, `pearson`).
- `--feature_selection`: El método de selección de características (por ejemplo, `clustering`).
- `--k`: El número de características a seleccionar.
- `--model`: El modelo de aprendizaje automático a utilizar (por ejemplo, `nnet`).
- `--model_params`: Una cadena JSON que especifica los parámetros del modelo.
- `--radiomics_file`: El archivo que contiene las características radiómicas.
- `--split_file`: El archivo CSV que contiene la división de entrenamiento y prueba.
- `--output_folder`: El directorio de salida para los resultados.

### Ejemplo

```bash
python main.py --correlation_method pearson --feature_selection clustering --k 5 --model nnet --model_params '{"architecture": [8], "alpha": 1}' --radiomics_file ".\data\radiomics_features_nodule_patch.csv" --split_file ".\data\train_10cv.csv" --output_folder ".\data"
```