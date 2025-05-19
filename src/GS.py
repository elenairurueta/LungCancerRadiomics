from Imports import *
from Statistics import pearson_correlation
from Data import read_csv


def GS(feature_selection_method, model_function, hyperparameter_grid, out_path='C:\\dev\\LungCancerRadiomics\\data', split_csv=None):
    """
    Función genérica para realizar Grid Search.

    Parámetros:
    - feature_selection_method: Función para seleccionar características.
    - model_function: Función del modelo a entrenar.
    - hyperparameter_grid: Diccionario con los hiperparámetros a optimizar.
    - outPath: Ruta base para los archivos de entrada y salida.
    """
    inputCSV = os.path.join(out_path, 'radiomics.csv')
    outputRadiomicsFilepath = os.path.join(out_path, 'radiomics_features_nodule_patch.csv')
    outputFilteringFilepath = os.path.join(out_path, 'correlation_log.txt')
    outputSelectionFilepath = os.path.join(out_path, 'feature_selection_log.txt')
    outputModelFilepath = os.path.join(out_path, 'trained_models')

    radiomics_count, radiomics_features = read_csv(outputRadiomicsFilepath)
    print('Cantidad de features seleccionadas por pyRadiomics:', radiomics_count[1])
    radiomics_features.drop(columns=['PatientID','StudyDate','CoordX','CoordY','CoordZ','LesionID','NoduleID','Age_at_StudyDate','Gender','SPLIT','TimeStep','FOLD','image','mask'], inplace=True, errors='ignore')

    filtered_features = pearson_correlation(radiomics_features, outPath=outputFilteringFilepath)
    print('Características filtradas:', len(filtered_features))

    hyperparameter_combinations = list(product(*hyperparameter_grid.values()))
    best_hyp = None
    best_score = -float('inf')
    results = []

    for combination in hyperparameter_combinations:
        hyperparameters = dict(zip(hyperparameter_grid.keys(), combination))

        k = hyperparameters.pop('k', None)
        if('alpha' in hyperparameters):
            alpha = hyperparameters.pop('alpha', None)
            selected_features = feature_selection_method(filtered_features, outPath=outputSelectionFilepath, k=k, alpha=alpha)
        else:
            selected_features = feature_selection_method(filtered_features, outPath=outputSelectionFilepath, k=k)
        print(f'Características seleccionadas con k={k}:', selected_features)

        print(f"Entrenando modelo con {hyperparameters}")
        avg_scores, std_scores = model_function(selected_features, outPath_model=outputModelFilepath,
                                outPath_log=os.path.join(outputModelFilepath, 'model_training_log.txt'), **hyperparameters, split_csv=split_csv)
        results.append({**hyperparameters, 'mean_roc_auc': avg_scores['roc_auc'], 'std_roc_auc': std_scores['roc_auc']})

        if 'fit_time' in avg_scores:
            print("Tiempo de ajuste promedio: ", avg_scores['fit_time'])
        if 'score_time' in avg_scores:
            print("Tiempo de evaluación promedio: ", avg_scores['score_time'])

        for score in avg_scores:
            if score not in ('fit_time', 'score_time', 'estimator'):
                print(f"{score}: " + "(%0.3f ± %0.3f)" % (avg_scores[score], std_scores[score]))

        if avg_scores['roc_auc'] > best_score:
            best_score = avg_scores['roc_auc']
            best_std = std_scores['roc_auc']
            best_hyp = {**hyperparameters, 'k': k}

    print('Entrenamiento del modelo completo.')
    print("\nMejores Hiperparámetros:")
    for key, value in best_hyp.items():
        print(f"{key}: {value}")
    print(f"Mejor ROC AUC Promedio: {best_score:.3f} ± {best_std:.3f}")
    return best_hyp, best_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid Search for model training.")
    parser.add_argument("--feature_selection", type=str, required=True, help="Feature selection method (e.g., 'pca').")
    parser.add_argument("--model", type=str, required=True, help="Model to train (e.g., 'random_forest').")
    parser.add_argument("--hyperparameters", type=str, required=True, help="Hyperparameter grid as a JSON string.")
    parser.add_argument("--out_path", type=str, default="C:\\dev\\LungCancerRadiomics\\data", help="Output path for files.")
    parser.add_argument("--split_csv_path", type=str, default=None, help="Path to CSV file for splitting data.", required=False)

    args = parser.parse_args()

    from Statistics import pca_feature_selection, anova_feature_selection, sfm_feature_selection, clustering_feature_selection, lasso_feature_selection, sfs_feature_selection, rfe_feature_selection, rfecv_feature_selection
    from Model import model_random_forest, model_bagging, model_nnet, model_knn, model_xgboost, model_svm

    feature_selection_methods = {
        "pca": pca_feature_selection,
        "anova": anova_feature_selection,
        "sfm": sfm_feature_selection,
        "clustering": clustering_feature_selection,
        "lasso": lasso_feature_selection,
        "sfs": sfs_feature_selection,
        "rfe": rfe_feature_selection,
        "rfecv": rfecv_feature_selection
    }

    model_functions = {
        "random_forest": model_random_forest,
        "bagging": model_bagging,
        "nnet": model_nnet,
        "knn": model_knn,
        "xgboost": model_xgboost,
        "svm": model_svm
    }

    feature_selection_method = feature_selection_methods.get(args.feature_selection)
    model_function = model_functions.get(args.model)

    if not feature_selection_method or not model_function:
        raise ValueError("Invalid feature selection method or model specified.")

    import json
    hyperparameter_grid = json.loads(args.hyperparameters)

    _,split_csv = read_csv(args.split_csv_path) if args.split_csv_path else None

    GS(feature_selection_method=feature_selection_method, model_function=model_function, hyperparameter_grid=hyperparameter_grid, out_path=args.out_path, split_csv=split_csv)