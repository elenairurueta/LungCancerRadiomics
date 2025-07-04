from Imports import *

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


def model_random_forest(features, outPath_model='', outPath_log='', n_trees=100, max_depth=3, min_samples_split=0.5, crossVal=True, split_csv=None):
    model = RandomForestClassifier(n_estimators=n_trees, 
                                max_depth=max_depth, 
                                min_samples_split=min_samples_split, 
                                oob_score=True, 
                                random_state=0, 
                                verbose=3
    )
    return train_model(features, model=model, outPath_model=outPath_model, outPath_log=outPath_log, crossVal=crossVal, split_csv=split_csv)

def model_bagging(features, outPath_model='', outPath_log='', n_estimators=10, max_samples=1.0, max_features=1.0, crossVal=True, split_csv=None):
    model = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=0),
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        bootstrap=True,
        random_state=0,
        verbose=3
    )
    return train_model(features, model=model, outPath_model=outPath_model, outPath_log=outPath_log, crossVal=crossVal, split_csv=split_csv)

def model_nnet(features, outPath_model='', outPath_log='', architecture=(8,), alpha=0.1, max_iter=600, crossVal=True, split_csv=None):
    model = MLPClassifier(
        hidden_layer_sizes=architecture,
        alpha=alpha,
        max_iter=max_iter,
        random_state=0,
        verbose=True
    )
    return train_model(features, model=model, outPath_model=outPath_model, outPath_log=outPath_log, crossVal=crossVal, split_csv=split_csv)

def model_knn(features, outPath_model='', outPath_log='', n_neighbors=5, crossVal=True, split_csv=None):
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights='uniform'
    )
    return train_model(features, model=model, outPath_model=outPath_model, outPath_log=outPath_log, crossVal=crossVal, split_csv=split_csv)

def model_xgboost(features, outPath_model='', outPath_log='', n_trees=100, max_depth=3, learning_rate=0.1, min_samples_leaf=1, crossVal=True, split_csv=None):
    model = GradientBoostingClassifier(
        n_estimators=n_trees, 
        learning_rate=learning_rate, 
        max_depth=max_depth, 
        min_samples_leaf=min_samples_leaf,
        random_state=0
    )
    return train_model(features, model=model, outPath_model=outPath_model, outPath_log=outPath_log, crossVal=crossVal, split_csv=split_csv)

def model_svm(features, outPath_model='', outPath_log='', kernel='linear', C=1.0, crossVal=True, split_csv=None):
    model = SVC(
        kernel=kernel,
        C=C,
        probability=True,
        random_state=0
    )
    return train_model(features, model=model, outPath_model=outPath_model, outPath_log=outPath_log, crossVal=crossVal, split_csv=split_csv)

def train_model(features, model, outPath_model='', outPath_log='', crossVal=True, scoring=None, cv=None, split_csv=None):
    """
    Función genérica para entrenar un modelo con validación cruzada automática o manual.

    Parámetros:
    - features: DataFrame con las características y etiquetas.
    - model: Modelo a entrenar (debe ser un estimador compatible con scikit-learn).
    - outPath_model: Ruta para guardar el modelo entrenado.
    - outPath_log: Ruta para guardar el log del entrenamiento.
    - crossVal: Si se realiza validación cruzada.
    - scoring: Lista de métricas para validación cruzada.
    - cv: Objeto de validación cruzada (por ejemplo, RepeatedStratifiedKFold).
    - split_csv: DataFrame con la columna `FOLD` para realizar validación cruzada manual.

    Retorna:
    - scores: Diccionario con las métricas de validación cruzada o vacío si no se realiza validación cruzada.
    """

    os.makedirs(os.path.dirname(outPath_log), exist_ok=True)
    with open(outPath_log, 'w') as log_file:
        log_file.write('')
    logging.basicConfig(filename=outPath_log, level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()
    original_stdout = sys.stdout

    exclude_columns = [col for col in features.columns if col in ['Image', 'Mask', 'SeriesInstanceUID', 'AnnotationID', 'image', 'mask', 'Label', 'label', 'FOLD'] or col.startswith('diagnostics_')]

    try:
        sys.stdout = StreamToLogger(logger, logging.INFO)
        logger.info(f"Entrenando modelo: {model}")
        
        if crossVal:
            if split_csv is not None:
                split_features = features.merge(split_csv, on=['hospital', 'subject'], how='left')
                drop_cols = ['PatientID','SeriesInstanceUID','StudyDate','CoordX','CoordY','CoordZ','LesionID','AnnotationID','NoduleID','Age_at_StudyDate','Gender','SPLIT','TimeStep', 'OS_12m', 'PFS_6m', 'hospital', 'subject', 'Image', 'Mask', 'Labels']
                split_features.drop(columns=drop_cols, inplace=True, errors='ignore')

                folds = split_features['FOLD'].unique()
                all_scores = {metric: [] for metric in (scoring or ['precision_macro', 'recall_macro', 'accuracy', 'roc_auc'])}

                for fold in folds:
                    logger.info(f"Procesando fold {fold}...")
                    from Data import split_by_fold
                    train_data, val_data = split_by_fold(split_features, fold_column='FOLD', current_fold=fold)

                    y_train = train_data['Label' if 'Label' in train_data.columns else 'label'].to_numpy()
                    X_train = train_data.drop(columns=exclude_columns, errors='ignore').to_numpy()
                    y_val = val_data['Label' if 'Label' in val_data.columns else 'label'].to_numpy()
                    X_val = val_data.drop(columns=exclude_columns, errors='ignore').to_numpy()

                    logger.info(f"Forma de X_train: {X_train.shape}, y_train: {y_train.shape}")
                    logger.info(f"Forma de X_val: {X_val.shape}, y_val: {y_val.shape}")

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    y_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None

                    for metric in all_scores.keys():
                        if metric == 'precision_macro':
                            all_scores[metric].append(precision_score(y_val, y_pred, average='macro', zero_division=0))
                        elif metric == 'recall_macro':
                            all_scores[metric].append(recall_score(y_val, y_pred, average='macro', zero_division=0))
                        elif metric == 'accuracy':
                            all_scores[metric].append(accuracy_score(y_val, y_pred))
                        elif metric == 'roc_auc':
                            all_scores[metric].append(roc_auc_score(y_val, y_prob))

                    fold_model_path = os.path.join(outPath_model, f'model_fold_{fold}.joblib')
                    dump(model, fold_model_path)
                    logger.info(f"Modelo del fold {fold} guardado en: {fold_model_path}")

                avg_scores = {metric: np.mean(values) for metric, values in all_scores.items()}
                std_scores = {metric: np.std(values) for metric, values in all_scores.items()}

                for metric in avg_scores:
                    logger.info(f"{metric}: {avg_scores[metric]:.4f} ± {std_scores[metric]:.4f}")

                return avg_scores, std_scores
            else:
                n_splits = 3
                if scoring is None:
                    scoring = {
                        'precision_macro': make_scorer(precision_score, average='macro', zero_division=0),
                        'recall_macro': make_scorer(recall_score, average='macro', zero_division=0),
                        'accuracy': 'accuracy',
                        'roc_auc': 'roc_auc'
                    }
                if cv is None:
                    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=2, random_state=0)
                filtered_features = features.drop(columns=exclude_columns, errors='ignore')
                X = filtered_features.to_numpy()
                y = features['Label' if 'Label' in features.columns else 'label'].to_numpy()
                scores = cross_validate(model, X, y, cv=cv, scoring=scoring, return_estimator=True)
                logger.info(f"Tiempo de ajuste: {scores['fit_time'].mean()}")
                logger.info(f"Tiempo de evaluación: {scores['score_time'].mean()}")
                for idx, estimator in enumerate(scores['estimator']):
                    repetition = idx // n_splits + 1 
                    fold = idx % n_splits + 1 
                    fold_model_path = os.path.join(outPath_model, f'model_rep_{repetition}_fold_{fold}.joblib')
                    dump(estimator, fold_model_path)
                    logger.info(f"Modelo de repetición {repetition}, fold {fold} guardado en: {fold_model_path}")
                for score in scores:
                    if score not in ('fit_time', 'score_time', 'estimator'):
                        logger.info(f"{score}: " + "(%0.2f ± %0.2f)" % (scores[score].mean(), scores[score].std()))

                avg_scores = {metric: np.mean(values) for metric, values in scores.items()}
                std_scores = {metric: np.std(values) for metric, values in scores.items()}
                return avg_scores, std_scores
            
        else:
            filtered_features = features.drop(columns=exclude_columns, errors='ignore')
            X = filtered_features.to_numpy()
            y = features['Label' if 'Label' in features.columns else 'label'].to_numpy()
            model.fit(X, y)
            dump(model, outPath_model)
            logger.info(f"Modelo guardado en: {outPath_model}")
            return {}
    finally:
        sys.stdout = original_stdout

def lazy_classifier(features, outPath_model='', outPath_log='', crossVal=True, split_csv=None):
    """
    Función para entrenar un modelo utilizando LazyClassifier.

    Parámetros:
    - features: DataFrame con las características y etiquetas.
    - outPath_model: Ruta para guardar el modelo entrenado.
    - outPath_log: Ruta para guardar el log del entrenamiento.
    - crossVal: Si se realiza validación cruzada.
    - split_csv: DataFrame con la columna `FOLD` para realizar validación cruzada manual.

    Retorna:
    - scores: Diccionario con las métricas de validación cruzada o vacío si no se realiza validación cruzada.
    """

    os.makedirs(os.path.dirname(outPath_log), exist_ok=True)
    with open(outPath_log, 'w') as log_file:
        log_file.write('')
    logging.basicConfig(filename=outPath_log, level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()
    original_stdout = sys.stdout

    exclude_columns = [col for col in features.columns if col in ['Image', 'Mask', 'SeriesInstanceUID', 'AnnotationID', 'image', 'mask', 'Label', 'label', 'FOLD'] or col.startswith('diagnostics_')]

    clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)

    try:
        sys.stdout = StreamToLogger(logger, logging.INFO)
        logger.info(f"Entrenando modelo: {clf}")
        
        if crossVal:
            if split_csv is not None:
                split_features = features.merge(split_csv, on=['SeriesInstanceUID', 'AnnotationID', 'label'], how='left')
                drop_cols = ['PatientID','SeriesInstanceUID','StudyDate','CoordX','CoordY','CoordZ','LesionID','AnnotationID','NoduleID','Age_at_StudyDate','Gender','SPLIT','TimeStep']
                split_features.drop(columns=drop_cols, inplace=True, errors='ignore')

                folds = split_features['FOLD'].unique()

                for fold in folds:
                    logger.info(f"Procesando fold {fold}...")
                    from Data import split_by_fold
                    train_data, val_data = split_by_fold(split_features, fold_column='FOLD', current_fold=fold)

                    y_train = train_data['Label' if 'Label' in train_data.columns else 'label'].to_numpy()
                    X_train = train_data.drop(columns=exclude_columns, errors='ignore').to_numpy()
                    y_val = val_data['Label' if 'Label' in val_data.columns else 'label'].to_numpy()
                    X_val = val_data.drop(columns=exclude_columns, errors='ignore').to_numpy()
                    
                    logger.info(f"Forma de X_train: {X_train.shape}, y_train: {y_train.shape}")
                    logger.info(f"Forma de X_val: {X_val.shape}, y_val: {y_val.shape}")

                    models,predictions = clf.fit(X_train, X_val, y_train, y_val)

                
                for model in models:
                    logger.info(model)

                return models, predictions
            else:
                n_splits = 3
                if scoring is None:
                    scoring = {
                        'precision_macro': make_scorer(precision_score, average='macro', zero_division=0),
                        'recall_macro': make_scorer(recall_score, average='macro', zero_division=0),
                        'accuracy': 'accuracy',
                        'roc_auc': 'roc_auc'
                    }
                if cv is None:
                    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=2, random_state=0)
                filtered_features = features.drop(columns=exclude_columns, errors='ignore')
                X = filtered_features.to_numpy()
                y = features['Label' if 'Label' in features.columns else 'label'].to_numpy()
                for train_idx, val_idx in cv.split(X, y):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    models, predictions = clf.fit(X_train, X_val, y_train, y_val)
                    logger.info(models, predictions)

                return models, predictions
            
        else:
            filtered_features = features.drop(columns=exclude_columns, errors='ignore')
            X = filtered_features.to_numpy()
            y = features['Label' if 'Label' in features.columns else 'label'].to_numpy()
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
            models, predictions = clf.fit(X_train, X_val, y_train, y_val)
            return models, predictions
        
    finally:
        sys.stdout = original_stdout
