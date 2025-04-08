from Imports import *
from Statistics import pearson_correlation, anova_ftest
from Radiomics import get_batch_radiomics

def main():

    outPath = 'C:\dev\LungCancerRadiomics\data'

    inputCSV = os.path.join(outPath, 'radiomics.csv')
    outputRadiomicsFilepath = os.path.join(outPath, 'radiomics_features.csv')
    outputFilteringFilepath = os.path.join(outPath, 'correlation_log.txt')
    outputSelectionFilepath = os.path.join(outPath, 'anova_log.txt')
    progress_filename = os.path.join(outPath, 'pyrad_log.txt')

    # params = {
    #     'binWidth': 25,
    #     'verbose': False,
    #     'sigma': [1, 2, 3],
    #     'wavelet': 'bior1.3',
    #     'level': 2
    # }
    params = os.path.join(outPath, 'Params.yaml')
    radiomics_count, radiomics_features = get_batch_radiomics(inputCSV, outputRadiomicsFilepath, progress_filename, params)
    print('Radiomics count:', radiomics_count)
    filtered_features = pearson_correlation(radiomics_features, outputFilteringFilepath)
    print('Filtered features:', filtered_features)
    selected_features = anova_ftest(filtered_features, outPath=outputSelectionFilepath, k=50)
    print('Selected features:', selected_features)

if __name__ == "__main__":
    main()