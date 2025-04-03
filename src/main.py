from Imports import *
from Radiomics import get_batch_radiomics

def main():

    outPath = 'C:\dev\LungCancerRadiomics\data'

    inputCSV = os.path.join(outPath, 'radiomics.csv')
    outputFilepath = os.path.join(outPath, 'radiomics_features.csv')
    progress_filename = os.path.join(outPath, 'pyrad_log.txt')

    # params = {
    #     'binWidth': 25,
    #     'verbose': False,
    #     'sigma': [1, 2, 3],
    #     'wavelet': 'bior1.3',
    #     'level': 2
    # }
    params = os.path.join(outPath, 'Params.yaml')
    radiomics_count = get_batch_radiomics(inputCSV, outputFilepath, progress_filename, params)
    print('Radiomics count:', radiomics_count)



if __name__ == "__main__":
    main()