from Imports import *
from Data import read_csv

def main():
    parser = argparse.ArgumentParser(description="Selección de características y entrenamiento de modelos para cáncer de pulmón utilizando radiomica.")
    
    # Argumentos para selección de características
    parser.add_argument('--correlation_method', type=str, choices=['pearson', 'spearman', 'kendall'], 
                        default='pearson', help="Método de correlación a usar: 'pearson'.")
    parser.add_argument('--feature_selection', type=str, choices=['anova', 'clustering', 'sfm', 'pca', 'lasso'], 
                        required=True, help="Método de selección de características: 'anova', 'clustering', 'sfm', 'pca' o 'lasso'.")
    parser.add_argument('--k', type=int, default=10, 
                        help="Número de características o clusters a seleccionar (por defecto: 10).")
    parser.add_argument('--alpha', type=int, default=0.001, required=False,
                        help="Valor de alpha para la selección de características (por defecto: 0.001).")
    
    # Argumentos para el modelo
    parser.add_argument('--model', type=str, choices=['svm', 'random_forest', 'xgboost', 'bagging', 'nnet', 'kNN'], 
                        required=True, help="Modelo a usar: 'svm', 'random_forest', 'xgboost', 'bagging', 'nnet' o 'kNN'.")
    parser.add_argument('--model_params', type=str, default='', 
                        help="Hiperparámetros del modelo en formato JSON (ejemplo: '{\"C\": 1.0, \"kernel\": \"linear\"}').")
    
    # Argumentos generales
    parser.add_argument('--radiomics_file', type=str, required=False, 
                        help="Ruta al archivo de entrada con las características.")
    parser.add_argument('--input_file', type=str, required=False, 
                        help="Ruta al archivo de entrada con las rutas a las imágenes y máscaras.")
    parser.add_argument('--split_file', type=str, required=False, 
                        help="Ruta al archivo que especifica split de folds para crossval.")
    parser.add_argument('--output_folder', type=str, required=True, 
                        help="Ruta a la carpeta de salida para guardar los resultados.")
    
    args = parser.parse_args()
    
    if args.radiomics_file:
        _, features = read_csv(args.radiomics_file)
        features.drop(columns=['PatientID','StudyDate','CoordX','CoordY','CoordZ','LesionID','NoduleID','Age_at_StudyDate','Gender','SPLIT','TimeStep','FOLD','image','mask'], inplace=True, errors='ignore')
    elif args.input_file:
        from Radiomics import get_batch_radiomics
        features = get_batch_radiomics(args.input_file, outPath=os.path.join(args.output_folder, 'radiomics_features.csv'), params=None)
    else:
        features = None


    if args.split_file:
        _, split_csv = read_csv(args.split_file)
    else:
        split_csv = None
    if args.correlation_method == 'pearson':
        from Statistics import pearson_correlation
        filtered_features = pearson_correlation(features, outPath=os.path.join(args.output_folder, 'correlation_log.txt'))
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
    elif args.feature_selection == 'lasso':
        from Statistics import lasso_feature_selection
        selected_features = lasso_feature_selection(filtered_features, k=args.k, alpha=args.alpha, outPath=os.path.join(args.output_folder, 'feature_selection_lasso_log.txt'))
    else:
        raise(ValueError("Método de selección de características no implementado."))
    
    print(f"Forma de las características seleccionadas: {selected_features.shape}")

    model_params = json.loads(args.model_params) if args.model_params else {}

    if args.model == 'svm':
        from Model import model_svm
        model = model_svm(selected_features, outPath_model=os.path.join(args.output_folder, 'models'), outPath_log=os.path.join(args.output_folder, 'model_SVM_training_log.txt'), **model_params, split_csv=split_csv)
    elif args.model == 'random_forest':
        from Model import model_random_forest
        scores = model_random_forest(selected_features, outPath_model=os.path.join(args.output_folder, 'models'), outPath_log=os.path.join(args.output_folder, 'model_RF_training_log.txt'), **model_params, split_csv=split_csv)
    elif args.model == 'xgboost':
        from Model import model_xgboost
        scores = model_xgboost(selected_features, outPath_model=os.path.join(args.output_folder, 'models'), outPath_log=os.path.join(args.output_folder, 'model_XGB_training_log.txt'), **model_params, split_csv=split_csv)
    elif args.model == 'bagging':
        from Model import model_bagging
        scores = model_bagging(selected_features, outPath_model=os.path.join(args.output_folder, 'models'), outPath_log=os.path.join(args.output_folder, 'model_BAG_training_log.txt'), **model_params, split_csv=split_csv)
    elif args.model == 'nnet':
        from Model import model_nnet
        scores = model_nnet(selected_features, outPath_model=os.path.join(args.output_folder, 'models'), outPath_log=os.path.join(args.output_folder, 'model_NNET_training_log.txt'), **model_params, split_csv=split_csv)
    elif args.model == 'kNN':   
        from Model import model_knn
        scores = model_knn(selected_features, outPath_model=os.path.join(args.output_folder, 'models'), outPath_log=os.path.join(args.output_folder, 'model_kNN_training_log.txt'), **model_params, split_csv=split_csv)
    else:
        raise NotImplementedError("Modelo no implementado.")

    filtered_scores = {metric: values for metric, values in scores.items() if metric not in ['estimator', 'fit_time', 'score_time']}    
    avg_scores = {metric: np.mean(values) for metric, values in filtered_scores.items()}
    std_scores = {metric: np.std(values) for metric, values in filtered_scores.items()}
    
    for metric in avg_scores:
        print(f"{metric}: {avg_scores[metric]:.4f} ± {std_scores[metric]:.4f}")

if __name__ == "__main__":
    main()