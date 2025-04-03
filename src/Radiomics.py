from __future__ import print_function
import os
import radiomics
from radiomics import featureextractor

import collections
import csv
import logging
import os

import SimpleITK as sitk


def main():
    testCase = '100004_T0'
    dataDir = '//10.5.38.120/BIT-UPM-projects/NODULES/SCRATCH_STUDENTS/examples/benign/100004'
    image_path = os.path.join(dataDir, testCase + '.nrrd')
    label_path = os.path.join(dataDir, testCase + '_seg.nrrd')

    params = {}
    params['binWidth'] = 20
    params['verbose'] = True

    extractor = featureextractor.RadiomicsFeatureExtractor(**params)
    extractor.enableAllImageTypes()
    print('Extraction parameters:\n\t', extractor.settings)
    print('Enabled filters:\n\t', extractor.enabledImagetypes)
    print('Enabled features:\n\t', extractor.enabledFeatures)

    result = extractor.execute(image_path, label_path)
    print(len(result))
    # for key, value in result.items():
    #     print('\t', key, ':', value)

def batch_processing():
    outPath = r'C:/dev/LungNodulesDetection/data'

    inputCSV = os.path.join(outPath, 'radiomics.csv')
    outputFilepath = os.path.join(outPath, 'radiomics_features.csv')
    progress_filename = os.path.join(outPath, 'pyrad_log.txt')
    #params = os.path.join(outPath, 'exampleSettings', 'Params.yaml')
    params = {}
    params['binWidth'] = 20
    params['verbose'] = True

    # Configure logging
    rLogger = logging.getLogger('radiomics')

    # Create handler for writing to log file
    handler = logging.FileHandler(filename=progress_filename, mode='w')
    handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
    rLogger.addHandler(handler)

    # Initialize logging for batch log messages
    logger = rLogger.getChild('batch')

    # Set verbosity level for output to stderr (default level = WARNING)
    radiomics.setVerbosity(logging.INFO)

    logger.info('pyradiomics version: %s', radiomics.__version__)
    logger.info('Loading CSV')

    flists = []
    try:
        with open(inputCSV, 'r') as inFile:
            cr = csv.DictReader(inFile, lineterminator='\n')
            flists = [row for row in cr]
    except Exception:
        logger.error('CSV READ FAILED', exc_info=True)

    logger.info('Loading Done')
    logger.info('Patients: %d', len(flists))

    if len(params)>0:
        extractor = featureextractor.RadiomicsFeatureExtractor(**params)
    else:  # Parameter file not found, use hardcoded settings instead
        settings = {}
        settings['binWidth'] = 25
        settings['resampledPixelSpacing'] = None  # [3,3,3]
        settings['interpolator'] = sitk.sitkBSpline
        settings['enableCExtensions'] = True

        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
        # extractor.enableInputImages(wavelet= {'level': 2})

    logger.info('Enabled input images types: %s', extractor.enabledImagetypes)
    logger.info('Enabled features: %s', extractor.enabledFeatures)
    logger.info('Current settings: %s', extractor.settings)

    headers = None

    for idx, entry in enumerate(flists, start=1):

        logger.info("(%d/%d) Processing Patient (Image: %s, Mask: %s)", idx, len(flists), entry['Image'], entry['Mask'])

        imageFilepath = entry['Image']
        maskFilepath = entry['Mask']
        label = entry.get('Label', None)

        if str(label).isdigit():
            label = int(label)
        else:
            label = None

        if (imageFilepath is not None) and (maskFilepath is not None):
            featureVector = collections.OrderedDict(entry)
            featureVector['Image'] = os.path.basename(imageFilepath)
            featureVector['Mask'] = os.path.basename(maskFilepath)

            try:
                featureVector.update(extractor.execute(imageFilepath, maskFilepath, label))

                with open(outputFilepath, 'a') as outputFile:
                    writer = csv.writer(outputFile, lineterminator='\n')
                    if headers is None:
                        headers = list(featureVector.keys())
                        writer.writerow(headers)

                    row = []
                    for h in headers:
                        row.append(featureVector.get(h, "N/A"))
                    writer.writerow(row)
            except Exception:
                logger.error('FEATURE EXTRACTION FAILED', exc_info=True)

batch_processing()
#main()

