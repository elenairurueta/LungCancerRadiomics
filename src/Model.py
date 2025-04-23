from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from Imports import *
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold
from joblib import dump
import os
import logging
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

class StreamToLogger:
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.line_buffer = ''

    def write(self, message):
        if message.strip():
            self.logger.log(self.log_level, message.strip())

    def flush(self):
        pass

def model_random_forest(features, outPath_model='', outPath_log='', n_trees=100, max_depth=3, min_samples_split=0.5, crossVal=True):

    outPath_model = os.path.join(outPath_model,f'model_RF_nt_{n_trees}_md_{max_depth}_mss_{int(min_samples_split*10)}.joblib')

    exclude_columns = [col for col in features.columns if col in ['Image', 'Mask', 'Label'] or col.startswith('diagnostics_')]
    filtered_features = features.drop(columns=exclude_columns, errors='ignore')
    
    X = filtered_features.to_numpy()
    y = features['Label'].to_numpy()

    os.makedirs(os.path.dirname(outPath_log), exist_ok=True)
    with open(outPath_log, 'w') as log_file:
        log_file.write('')
    logging.basicConfig(filename=outPath_log, level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()
    original_stdout = sys.stdout
    
    try:
        sys.stdout = StreamToLogger(logger, logging.INFO)
        rf = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth, min_samples_split=min_samples_split, 
                                    oob_score=True, random_state=0, verbose=3)
        logger.info(f"Entrenando modelo Random Forest con n_trees={n_trees}, max_depth={max_depth}, min_samples_split={min_samples_split}")
        if(crossVal):
            scoring = ['precision_macro', 'recall_macro', 'accuracy', 'roc_auc']
            cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=0)
            scores = cross_validate(rf, X, y, cv=cv, scoring=scoring)
            logger.info(f"Tiempo de ajuste: {scores['fit_time'].mean()}")
            logger.info(f"Tiempo de evaluación: {scores['score_time'].mean()}")
            for score in scores:
                if score not in ('fit_time', 'score_time'):
                    logger.info(f"{score}: " + "(%0.2f +- %0.2f)" % (scores[score].mean(), scores[score].std()))
        else:
            rf.fit(X, y)
            os.makedirs(os.path.dirname(outPath_model), exist_ok=True)
            scores = []
        dump(rf, outPath_model) 
    finally:
        sys.stdout = original_stdout

    return scores

def model_bagging(features, outPath_model='', outPath_log='', n_estimators=10, max_samples=1.0, max_features=1.0, crossVal=True):
    """
    Implementa Bagging para tareas de clasificación.

    Parámetros:
    - features: DataFrame de características.
    - outPath_model: Ruta para guardar el modelo entrenado.
    - outPath_log: Ruta para guardar el log del entrenamiento.
    - n_estimators: Número de estimadores base en el conjunto.
    - max_samples: Fracción de muestras para cada estimador base.
    - max_features: Fracción de características para cada estimador base.
    - crossVal: Si se realiza validación cruzada.

    Retorna:
    - Puntuaciones de validación cruzada.
    """
    outPath_model = os.path.join(outPath_model, f'model_BAG_ne_{n_estimators}_ms_{int(max_samples*100)}_mf_{int(max_features*100)}.joblib')

    exclude_columns = [col for col in features.columns if col in ['Image', 'Mask', 'Label'] or col.startswith('diagnostics_')]
    filtered_features = features.drop(columns=exclude_columns, errors='ignore')

    X = filtered_features.to_numpy()
    y = features['Label'].to_numpy()

    os.makedirs(os.path.dirname(outPath_log), exist_ok=True)
    with open(outPath_log, 'w') as log_file:
        log_file.write('')
    logging.basicConfig(filename=outPath_log, level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()
    original_stdout = sys.stdout

    try:
        sys.stdout = StreamToLogger(logger, logging.INFO)

        estimator = DecisionTreeClassifier(random_state=0)
        model = BaggingClassifier(estimator=estimator, n_estimators=n_estimators, max_samples=max_samples, 
                                  max_features=max_features, bootstrap=True, random_state=0)

        logger.info(f"Entrenando modelo Bagging con n_estimators={n_estimators}, max_samples={max_samples}, max_features={max_features}")

        if crossVal:
            scoring = ['precision_macro', 'recall_macro', 'accuracy', 'roc_auc']
            cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=0)
            scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
            logger.info(f"Tiempo de ajuste: {scores['fit_time'].mean()}")
            logger.info(f"Tiempo de evaluación: {scores['score_time'].mean()}")
            for score in scores:
                if score not in ('fit_time', 'score_time'):
                    logger.info(f"{score}: " + "(%0.2f +- %0.2f)" % (scores[score].mean(), scores[score].std()))
        else:
            model.fit(X, y)
            os.makedirs(os.path.dirname(outPath_model), exist_ok=True)
            scores = []
        dump(model, outPath_model)
    finally:
        sys.stdout = original_stdout

    return scores

def model_nnet(features, outPath_model='', outPath_log='', architecture=(8,), alpha=0.1, max_iter=600, crossVal=True):
    """
    Implementa una red neuronal (NNET) para tareas de clasificación.

    Parámetros:
    - features: DataFrame de características.
    - outPath_model: Ruta para guardar el modelo entrenado.
    - outPath_log: Ruta para guardar el log del entrenamiento.
    - architecture: Tupla que define los tamaños de las capas ocultas (por ejemplo, (8,), (8, 8), (16, 16)).
    - alpha: Parámetro de regularización L2.
    - max_iter: Número máximo de iteraciones para el entrenamiento.
    - crossVal: Si se realiza validación cruzada.

    Retorna:
    - Puntuaciones de validación cruzada.
    """
    outPath_model = os.path.join(outPath_model, f'model_NNET_arch_{architecture}_alpha_{alpha}.joblib')

    exclude_columns = [col for col in features.columns if col in ['Image', 'Mask', 'Label'] or col.startswith('diagnostics_')]
    filtered_features = features.drop(columns=exclude_columns, errors='ignore')

    X = filtered_features.to_numpy()
    y = features['Label'].to_numpy()

    os.makedirs(os.path.dirname(outPath_log), exist_ok=True)
    with open(outPath_log, 'w') as log_file:
        log_file.write('')
    logging.basicConfig(filename=outPath_log, level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()
    original_stdout = sys.stdout

    try:
        sys.stdout = StreamToLogger(logger, logging.INFO)

        model = MLPClassifier(hidden_layer_sizes=architecture, alpha=alpha, max_iter=max_iter, random_state=0)

        logger.info(f"Entrenando modelo NNET con arquitectura={architecture}, alpha={alpha}, max_iter={max_iter}")

        if crossVal:
            scoring = ['precision_macro', 'recall_macro', 'accuracy', 'roc_auc']
            cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=0)
            scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
            logger.info(f"Tiempo de ajuste: {scores['fit_time'].mean()}")
            logger.info(f"Tiempo de evaluación: {scores['score_time'].mean()}")
            for score in scores:
                if score not in ('fit_time', 'score_time'):
                    logger.info(f"{score}: " + "(%0.2f +- %0.2f)" % (scores[score].mean(), scores[score].std()))
        else:
            model.fit(X, y)
            os.makedirs(os.path.dirname(outPath_model), exist_ok=True)
            scores = []
        dump(model, outPath_model)
    finally:
        sys.stdout = original_stdout

    return scores

def model_knn(features, outPath_model='', outPath_log='', n_neighbors=5, crossVal=True):
    """
    Implementa K-Nearest Neighbors (KNN) para tareas de clasificación.

    Parámetros:
    - features: DataFrame de características.
    - outPath_model: Ruta para guardar el modelo entrenado.
    - outPath_log: Ruta para guardar el log del entrenamiento.
    - n_neighbors: Número de vecinos a considerar para la clasificación.
    - crossVal: Si se realiza validación cruzada.

    Retorna:
    - Puntuaciones de validación cruzada.
    """
    outPath_model = os.path.join(outPath_model, f'model_KNN_n_{n_neighbors}.joblib')

    exclude_columns = [col for col in features.columns if col in ['Image', 'Mask', 'Label'] or col.startswith('diagnostics_')]
    filtered_features = features.drop(columns=exclude_columns, errors='ignore')

    X = filtered_features.to_numpy()
    y = features['Label'].to_numpy()

    os.makedirs(os.path.dirname(outPath_log), exist_ok=True)
    with open(outPath_log, 'w') as log_file:
        log_file.write('')
    logging.basicConfig(filename=outPath_log, level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()
    original_stdout = sys.stdout

    try:
        sys.stdout = StreamToLogger(logger, logging.INFO)

        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights='uniform')

        logger.info(f"Entrenando modelo KNN con n_neighbors={n_neighbors}")

        if crossVal:
            scoring = ['precision_macro', 'recall_macro', 'accuracy', 'roc_auc']
            cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=0)
            scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
            logger.info(f"Tiempo de ajuste: {scores['fit_time'].mean()}")
            logger.info(f"Tiempo de evaluación: {scores['score_time'].mean()}")
            for score in scores:
                if score not in ('fit_time', 'score_time'):
                    logger.info(f"{score}: " + "(%0.2f +- %0.2f)" % (scores[score].mean(), scores[score].std()))
        else:
            model.fit(X, y)
            os.makedirs(os.path.dirname(outPath_model), exist_ok=True)
            scores = []
        dump(model, outPath_model)
    finally:
        sys.stdout = original_stdout

    return scores

def model_xgboost(features, outPath_model='', outPath_log='', n_estimators=100, max_depth=3, learning_rate=0.1, crossVal=True):
    return

def model_svm(features, outPath_model='', outPath_log='', kernel='linear', C=1.0, crossVal=True):
    return



