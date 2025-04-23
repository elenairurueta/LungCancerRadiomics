from Imports import *
from Statistics import pearson_correlation, anova_feature_selection, clustering_feature_selection, sfm_feature_selection, pca_feature_selection
from Radiomics import get_batch_radiomics, read_radiomics_csv
from Model import model_random_forest, model_bagging, model_nnet, model_knn

def GS_rf():

    outPath = 'C:\dev\LungCancerRadiomics\data'
    inputCSV = os.path.join(outPath, 'radiomics.csv')
    outputRadiomicsFilepath = os.path.join(outPath, 'radiomics_features.csv')
    outputProgressFilename = os.path.join(outPath, 'pyrad_log.txt')
    outputFilteringFilepath = os.path.join(outPath, 'correlation_log.txt')
    outputSelectionFilepath = os.path.join(outPath, 'feature_selection_log.txt')
    outputModelFilepath = os.path.join(outPath, 'trained_models')

    # params = os.path.join(outPath, 'Params.yaml')
    # radiomics_count, radiomics_features = get_batch_radiomics(inputCSV, params=params,
    #                                                          outPath=outputRadiomicsFilepath, progress_filename=outputProgressFilename)
    radiomics_count, radiomics_features = read_radiomics_csv(outputRadiomicsFilepath)
    print('Cantidad de features seleccionadas por pyRadiomics:', radiomics_count)
    filtered_features = pearson_correlation(radiomics_features, outPath=outputFilteringFilepath)
    print('Características filtradas:', filtered_features)

    n_trees_options = [50,100]
    max_depth_options = [1,2,3]
    min_samples_split_options = [0.1,0.5]
    k_options = [5,10,20,40]
    best_hyp = None
    best_score = -float('inf')
    results = []
    
    for n_trees, max_depth, min_samples_split, k in product(n_trees_options, max_depth_options, min_samples_split_options, k_options):

        #selected_features = anova_ftest(filtered_features, outPath=outputSelectionFilepath, k=k)
        #selected_features = hierarchical_clustering_features(filtered_features, outPath=outputSelectionFilepath, k=k)
        #selected_features = sfm_feature_selection(filtered_features, outPath=outputSelectionFilepath, k=k)
        selected_features = pca_feature_selection(filtered_features, outPath=outputSelectionFilepath, k=k)
        print(f'Características seleccionadas con k={k}:', selected_features)
        

        print(f"Entrenando modelo con n_trees={n_trees}, max_depth={max_depth}, min_samples_split={min_samples_split}")

        scores = model_random_forest(selected_features, 
                            n_trees=n_trees, max_depth=max_depth, min_samples_split=min_samples_split, 
                            outPath_model=outputModelFilepath,
                            outPath_log=os.path.join(outputModelFilepath,'model_RF_training_log.txt'))
        
        mean_roc_auc = scores['test_roc_auc'].mean()
        results.append({
            'n_trees': n_trees,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'k': k,
            'mean_roc_auc': mean_roc_auc
        })
        
        print("Tiempo de ajuste: ", scores['fit_time'].mean())
        print("Tiempo de evaluación: ", scores['score_time'].mean())
        for score in scores:
            if score not in ('fit_time', 'score_time'):
                print(f"{score}: " + "(%0.2f +- %0.2f)" % (scores[score].mean(), scores[score].std()))
    
        if mean_roc_auc > best_score:
            best_score = mean_roc_auc
            best_hyp = {
                'n_trees': n_trees,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'k': k
            }
    print('Entrenamiento del modelo completo.')
    print("\nMejores Hiperparámetros:")
    print(f"n_trees: {best_hyp['n_trees']}, max_depth: {best_hyp['max_depth']}, min_samples_split: {best_hyp['min_samples_split']}, k: {best_hyp['k']}")
    print(f"Mejor ROC AUC Promedio: {best_score:.2f}")


def GS_bagging():
    outPath = 'C:\dev\LungCancerRadiomics\data'
    inputCSV = os.path.join(outPath, 'radiomics.csv')
    outputRadiomicsFilepath = os.path.join(outPath, 'radiomics_features.csv')
    outputProgressFilename = os.path.join(outPath, 'pyrad_log.txt')
    outputFilteringFilepath = os.path.join(outPath, 'correlation_log.txt')
    outputSelectionFilepath = os.path.join(outPath, 'feature_selection_log.txt')
    outputModelFilepath = os.path.join(outPath, 'trained_models')

    # params = os.path.join(outPath, 'Params.yaml') 
    # radiomics_count, radiomics_features = get_batch_radiomics(inputCSV, params=params,
    #                                                          outPath=outputRadiomicsFilepath, progress_filename=outputProgressFilename)
    radiomics_count, radiomics_features = read_radiomics_csv(outputRadiomicsFilepath)
    print('Cantidad de features seleccionadas por pyRadiomics:', radiomics_count)
    filtered_features = pearson_correlation(radiomics_features, outPath=outputFilteringFilepath)
    print('Características filtradas:', filtered_features)

    n_estimators_options = [50,100]
    max_samples_options = [1,2,3]
    max_features_options = [0.1,0.5,1.0]
    k_options = [5,10,20,40]
    best_hyp = None
    best_score = -float('inf')
    results = []
    
    for n_estimators, max_samples, max_features, k in product(n_estimators_options, max_samples_options, max_features_options, k_options):

        #selected_features = anova_feature_selection(filtered_features, outPath=outputSelectionFilepath, k=k)
        #selected_features = clustering_feature_selection(filtered_features, outPath=outputSelectionFilepath, k=k)
        #selected_features = sfm_feature_selection(filtered_features, outPath=outputSelectionFilepath, k=k)
        selected_features = pca_feature_selection(filtered_features, outPath=outputSelectionFilepath, k=k)
        print(f'Características seleccionadas con k={k}:', selected_features)
        

        print(f"Entrenando modelo con n_estimators={n_estimators}, max_samples={max_samples}, max_features={max_features}")

        scores = model_bagging(selected_features,
                            n_estimators=n_estimators, max_samples=max_samples, max_features=max_features, 
                            outPath_model=outputModelFilepath,
                            outPath_log=os.path.join(outputModelFilepath,'model_Bagging_training_log.txt'))
        
        mean_roc_auc = scores['test_roc_auc'].mean()
        results.append({
            'n_estimators': n_estimators,
            'max_samples': max_samples,
            'max_features': max_features,
            'k': k,
            'mean_roc_auc': mean_roc_auc
        })
        
        print("Tiempo de ajuste: ", scores['fit_time'].mean())
        print("Tiempo de evaluación: ", scores['score_time'].mean())
        for score in scores:
            if score not in ('fit_time', 'score_time'):
                print(f"{score}: " + "(%0.2f +- %0.2f)" % (scores[score].mean(), scores[score].std()))
    
        if mean_roc_auc > best_score:
            best_score = mean_roc_auc
            best_hyp = {
                'n_estimators': n_estimators,
                'max_samples': max_samples,
                'max_features': max_features,
                'k': k
            }
    print('Entrenamiento del modelo completo.')
    print("\nMejores Hiperparámetros:")
    print(f"n_estimators: {best_hyp['n_estimators']}, max_samples: {best_hyp['max_samples']}, max_features: {best_hyp['max_features']}, k: {best_hyp['k']}")
    print(f"Mejor ROC AUC Promedio: {best_score:.2f}")

def GS_nnet():
    outPath = 'C:\dev\LungCancerRadiomics\data'
    inputCSV = os.path.join(outPath, 'radiomics.csv')
    outputRadiomicsFilepath = os.path.join(outPath, 'radiomics_features.csv')
    outputProgressFilename = os.path.join(outPath, 'pyrad_log.txt')
    outputFilteringFilepath = os.path.join(outPath, 'correlation_log.txt')
    outputSelectionFilepath = os.path.join(outPath, 'feature_selection_log.txt')
    outputModelFilepath = os.path.join(outPath, 'trained_models')

    # params = os.path.join(outPath, 'Params.yaml')
    # radiomics_count, radiomics_features = get_batch_radiomics(inputCSV, params=params,
    #                                                          outPath=outputRadiomicsFilepath, progress_filename=outputProgressFilename)
    radiomics_count, radiomics_features = read_radiomics_csv(outputRadiomicsFilepath)
    print('Cantidad de features seleccionadas por pyRadiomics:', radiomics_count)
    filtered_features = pearson_correlation(radiomics_features, outPath=outputFilteringFilepath)
    print('Características filtradas:', filtered_features)

    architecture_options = [(8,), (16,), (8,8), (16,16)]
    alpha_options = [1, 0.1]
    k_options = [5,10,20,40]
    best_hyp = None
    best_score = -float('inf')
    results = []
    
    for architecture, alpha, k in product(architecture_options, alpha_options, k_options):

        #selected_features = anova_feature_selection(filtered_features, outPath=outputSelectionFilepath, k=k)
        #selected_features = clustering_feature_selection(filtered_features, outPath=outputSelectionFilepath, k=k)
        #selected_features = sfm_feature_selection(filtered_features, outPath=outputSelectionFilepath, k=k)
        selected_features = pca_feature_selection(filtered_features, outPath=outputSelectionFilepath, k=k)
        print(f'Características seleccionadas con k={k}:', selected_features)
        

        print(f"Entrenando modelo con arquitectura={architecture}, alpha={alpha}")

        scores = model_nnet(selected_features,
                            architecture=architecture, alpha=alpha,
                            outPath_model=outputModelFilepath,
                            outPath_log=os.path.join(outputModelFilepath,'model_NN_training_log.txt'))
        
        mean_roc_auc = scores['test_roc_auc'].mean()
        results.append({
            'architecture': architecture,
            'alpha': alpha,
            'k': k,
            'mean_roc_auc': mean_roc_auc
        })
        
        print("Tiempo de ajuste: ", scores['fit_time'].mean())
        print("Tiempo de evaluación: ", scores['score_time'].mean())
        for score in scores:
            if score not in ('fit_time', 'score_time'):
                print(f"{score}: " + "(%0.2f +- %0.2f)" % (scores[score].mean(), scores[score].std()))
    
        if mean_roc_auc > best_score:
            best_score = mean_roc_auc
            best_hyp = {
                'architecture': architecture,
                'alpha': alpha,
                'k': k
            }
    print('Entrenamiento del modelo completo.')
    print("\nMejores Hiperparámetros:")
    print(f"Arquitectura: {best_hyp['architecture']}, Alpha: {best_hyp['alpha']}, k: {best_hyp['k']}")
    print(f"Mejor ROC AUC Promedio: {best_score:.2f}")

def GS_knn():
    outPath = 'C:\dev\LungCancerRadiomics\data'
    inputCSV = os.path.join(outPath, 'radiomics.csv')
    outputRadiomicsFilepath = os.path.join(outPath, 'radiomics_features.csv')
    outputProgressFilename = os.path.join(outPath, 'pyrad_log.txt')
    outputFilteringFilepath = os.path.join(outPath, 'correlation_log.txt')
    outputSelectionFilepath = os.path.join(outPath, 'feature_selection_log.txt')
    outputModelFilepath = os.path.join(outPath, 'trained_models')

    #params = os.path.join(outPath, 'Params.yaml')
    # radiomics_count, radiomics_features = get_batch_radiomics(inputCSV, params=params,
    #                                                          outPath=outputRadiomicsFilepath, progress_filename=outputProgressFilename)
    radiomics_count, radiomics_features = read_radiomics_csv(outputRadiomicsFilepath)
    print('Cantidad de radiomics:', radiomics_count)
    filtered_features = pearson_correlation(radiomics_features, outPath=outputFilteringFilepath)
    print('Características filtradas:', len(filtered_features))

    n_neighbors_options = [5,10,15,20] #[50,100,200] tienen que ser menores al numero de muestras
    k_options = [5,10,20,40]
    best_hyp = None
    best_score = -float('inf')
    results = []
    
    for n_neighbors, k in product(n_neighbors_options, k_options):

        selected_features = anova_feature_selection(filtered_features, outPath=outputSelectionFilepath, k=k)
        #selected_features = clustering_feature_selection(filtered_features, outPath=outputSelectionFilepath, k=k)
        #selected_features = sfm_feature_selection(filtered_features, outPath=outputSelectionFilepath, k=k)
        #selected_features = pca_feature_selection(filtered_features, outPath=outputSelectionFilepath, k=k)
        print(f'Características seleccionadas con k={k}:', selected_features)
        

        print(f"Entrenando modelo con n_neighbors={n_neighbors}")

        scores = model_knn(selected_features,
                            n_neighbors=n_neighbors, 
                            outPath_model=outputModelFilepath,
                            outPath_log=os.path.join(outputModelFilepath,'model_KNN_training_log.txt'))
        mean_roc_auc = scores['test_roc_auc'].mean()
        results.append({
            'n_neighbors': n_neighbors,
            'k': k,
            'mean_roc_auc': mean_roc_auc
        })
        
        print("Tiempo de ajuste: ", scores['fit_time'].mean())
        print("Tiempo de evaluación: ", scores['score_time'].mean())
        for score in scores:
            if score not in ('fit_time', 'score_time'):
                print(f"{score}: " + "(%0.2f +- %0.2f)" % (scores[score].mean(), scores[score].std()))
    
        if mean_roc_auc > best_score:
            best_score = mean_roc_auc
            best_hyp = {
                'n_neighbors': n_neighbors,
                'k': k
            }
    print('Entrenamiento del modelo completo.')
    print("\nMejores Hiperparámetros:")
    print(f"n_neighbors: {best_hyp['n_neighbors']}, k: {best_hyp['k']}")
    print(f"Mejor ROC AUC Promedio: {best_score:.2f}")

if __name__ == "__main__":
    GS_knn()