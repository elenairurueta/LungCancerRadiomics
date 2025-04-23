from Imports import *
from Radiomics import get_batch_radiomics, read_radiomics_csv


def main():
    parser = argparse.ArgumentParser(description="Selección de características y entrenamiento de modelos para cáncer de pulmón utilizando radiomica.")
    
    # Argumentos para selección de características
    parser.add_argument('--correlation_method', type=str, choices=['pearson', 'spearman', 'kendall'], 
                        default='pearson', help="Método de correlación a usar: 'pearson'.")
    parser.add_argument('--feature_selection', type=str, choices=['anova', 'clustering', 'sfm', 'pca'], 
                        required=True, help="Método de selección de características: 'anova', 'clustering', 'sfm' o 'pca'.")
    parser.add_argument('--k', type=int, default=10, 
                        help="Número de características o clusters a seleccionar (por defecto: 10).")
    
    # Argumentos para el modelo
    parser.add_argument('--model', type=str, choices=['svm', 'random_forest', 'xgboost', 'bagging', 'nnet', 'kNN'], 
                        required=True, help="Modelo a usar: 'svm', 'random_forest', 'xgboost', 'bagging', 'nnet' o 'kNN'.")
    parser.add_argument('--model_params', type=str, default='', 
                        help="Hiperparámetros del modelo en formato JSON (ejemplo: '{\"C\": 1.0, \"kernel\": \"linear\"}').")
    
    # Argumentos generales
    parser.add_argument('--input_file', type=str, required=True, 
                        help="Ruta al archivo de entrada con las características.")
    parser.add_argument('--output_folder', type=str, required=True, 
                        help="Ruta a la carpeta de salida para guardar los resultados.")
    
    args = parser.parse_args()
    
    try:
        _, features = read_radiomics_csv(args.input_file)
        if features.shape[1] == 3:
            raise FileNotFoundError("El archivo CSV no contiene datos válidos.")
    except FileNotFoundError:
        _, features = get_batch_radiomics(args.input_file, outPath=os.path.join(args.output_folder, 'radiomics_features.csv'), progress_filename=os.path.join(args.output_folder, 'pyrad_log.txt'), params=os.path.join(args.output_folder, 'Params.yaml'))
    
    if args.correlation_method == 'pearson':
        from Statistics import pearson_correlation
        filtered_features = pearson_correlation(features, outPath=os.path.join(args.output_folder, 'correlation_log.txt'))
    elif args.correlation_method == 'spearman':
        raise NotImplementedError("La correlación de Spearman no está implementada aún.")
        # from Statistics import spearman_correlation
        # filtered_features = spearman_correlation(features, outPath=os.path.join(args.output_folder, 'correlation_log.txt'))
    elif args.correlation_method == 'kendall':
        raise NotImplementedError("La correlación de Kendall no está implementada aún.")
        # from Statistics import kendall_correlation
        # filtered_features = kendall_correlation(features, outPath=os.path.join(args.output_folder, 'correlation_log.txt'))
    else:
        filtered_features = features
    
    print(f"Forma de las características filtradas: {filtered_features.shape}")

    if args.feature_selection == 'anova':
        from Statistics import anova_feature_selection
        selected_features = anova_feature_selection(filtered_features, outPath=os.path.join(args.output_folder, 'feature_selection_anova_log.txt'), k=args.k)
    elif args.feature_selection == 'clustering':
        from Statistics import clustering_feature_selection
        selected_features = clustering_feature_selection(filtered_features, k=args.k, outPath=os.path.join(args.output_folder, 'feature_selection_cluster_log.txt'))
    elif args.feature_selection == 'sfm':
        from Statistics import sfm_feature_selection
        selected_features = sfm_feature_selection(filtered_features, k=args.k, outPath=os.path.join(args.output_folder, 'feature_selection_sfm_log.txt'))
    elif args.feature_selection == 'pca':
        from Statistics import pca_feature_selection
        selected_features = pca_feature_selection(filtered_features, k=args.k, outPath=os.path.join(args.output_folder, 'feature_selection_pca_log.txt'))
    else:
        raise(ValueError("Método de selección de características no implementado."))
    
    print(f"Forma de las características seleccionadas: {selected_features.shape}")

    model_params = json.loads(args.model_params) if args.model_params else {}

    if args.model == 'svm':
        raise NotImplementedError("El modelo SVM no está implementado aún.")
        # from Model import model_svm
        # model = model_svm(**model_params)
    elif args.model == 'random_forest':
        from Model import model_random_forest
        scores = model_random_forest(selected_features, outPath_model=os.path.join(args.output_folder, 'trained_models'), outPath_log=os.path.join(args.output_folder, 'model_RF_training_log.txt'), **model_params)
    elif args.model == 'xgboost':
        raise NotImplementedError("El modelo XGBoost no está implementado aún.") 
        # from Model import model_xgboost
        # scores = model_xgboost(**model_params)
    elif args.model == 'bagging':
        from Model import model_bagging
        scores = model_bagging(selected_features, outPath_model=os.path.join(args.output_folder, 'trained_models'), outPath_log=os.path.join(args.output_folder, 'model_BAG_training_log.txt'), **model_params)
    elif args.model == 'nnet':
        from Model import model_nnet
        scores = model_nnet(selected_features, outPath_model=os.path.join(args.output_folder, 'trained_models'), outPath_log=os.path.join(args.output_folder, 'model_NNET_training_log.txt'), **model_params)
    elif args.model == 'kNN':   
        from Model import model_knn
        scores = model_knn(selected_features, outPath_model=os.path.join(args.output_folder, 'trained_models'), outPath_log=os.path.join(args.output_folder, 'model_kNN_training_log.txt'), **model_params)
    else:
        raise NotImplementedError("Modelo no implementado.")
    avg_scores = {metric: np.mean(values) for metric, values in scores.items()}
    std_scores = {metric: np.std(values) for metric, values in scores.items()}
    for metric in avg_scores:
        print(f"{metric}: {avg_scores[metric]:.4f} ± {std_scores[metric]:.4f}")

if __name__ == "__main__":
    main()