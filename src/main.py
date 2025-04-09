from Imports import *
from Statistics import pearson_correlation, anova_ftest
from Radiomics import get_batch_radiomics, read_radiomics_csv
from Model import model_random_forest

def main():

    outPath = 'C:\dev\LungCancerRadiomics\data'
    inputCSV = os.path.join(outPath, 'radiomics.csv')
    outputRadiomicsFilepath = os.path.join(outPath, 'radiomics_features.csv')
    outputProgressFilename = os.path.join(outPath, 'pyrad_log.txt')
    outputFilteringFilepath = os.path.join(outPath, 'correlation_log.txt')
    outputSelectionFilepath = os.path.join(outPath, 'anova_log.txt')
    outputModelFilepath = os.path.join(outPath, 'trained_models')

    # params = {
    #     'binWidth': 25,
    #     'verbose': False,
    #     'sigma': [1, 2, 3],
    #     'wavelet': 'bior1.3',
    #     'level': 2
    # }
    params = os.path.join(outPath, 'Params.yaml')
    
    #radiomics_count, radiomics_features = get_batch_radiomics(inputCSV, params=params,
    #                                                          outPath=outputRadiomicsFilepath, progress_filename=outputProgressFilename)
    radiomics_count, radiomics_features = read_radiomics_csv(outputRadiomicsFilepath)
    print('Radiomics count:', radiomics_count)
    filtered_features = pearson_correlation(radiomics_features, outPath=outputFilteringFilepath)
    print('Filtered features:', filtered_features)
    selected_features = anova_ftest(filtered_features, outPath=outputSelectionFilepath, k=50)
    print('Selected features:', selected_features)

    n_trees_options = [50,100]
    max_depth_options = [1,2,3]
    min_samples_split_options = [0.1,0.5]

    for n_trees, max_depth, min_samples_split in product(n_trees_options, max_depth_options, min_samples_split_options):
        print(f"Training model with n_trees={n_trees}, max_depth={max_depth}, min_samples_split={min_samples_split}")

        scores = model_random_forest(selected_features, 
                            n_trees=n_trees, max_depth=max_depth, min_samples_split=min_samples_split, 
                            outPath_model=os.path.join(outputModelFilepath,f'model_RF_nt_{n_trees}_md_{max_depth}_mss_{int(min_samples_split*10)}.joblib'),
                            outPath_log=os.path.join(outputModelFilepath,f'model_RF_training_log.txt'))
        
        print("Fit time: ", scores['fit_time'].mean())
        print("Score time: ", scores['score_time'].mean())
        for score in scores:
            if score not in ('fit_time', 'score_time'):
                print(f"{score}: " + "(%0.2f +- %0.2f)" % (scores[score].mean(), scores[score].std()))
    print('Model training complete.')

if __name__ == "__main__":
    main()