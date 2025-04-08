from Imports import *

def get_singleImage_radiomics(image_path, label_path, params=None, image_types=None, features=None):

    if params is None:
        params = {}
        params['binWidth'] = 25
        params['verbose'] = True   
    else:
        extractor = featureextractor.RadiomicsFeatureExtractor(**params)

    if image_types is not None:
        extractor.enableInputImages(**image_types)
    else:
        extractor.enableAllImageTypes()
    if features is not None:
        extractor.enableFeatures(**features)    
    else:
        extractor.enableAllFeatures()

    print('Extraction parameters:\n\t', extractor.settings)
    print('Enabled filters:\n\t', extractor.enabledImagetypes)
    print('Enabled features:\n\t', extractor.enabledFeatures)

    return extractor.execute(image_path, label_path)
    

def get_batch_radiomics(inputCSV, outputFilepath, progress_filename, params=None):    
    csv.field_size_limit(10**6)

    # Configure logging
    rLogger = logging.getLogger('radiomics')

    # Create handler for writing to log file
    handler = logging.FileHandler(filename=progress_filename, mode='w')
    handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
    rLogger.addHandler(handler)

    # Initialize logging for batch log messages
    logger = rLogger.getChild('batch')

    radiomics.setVerbosity(20)

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

    
    if params is None:
        params = {}
        params['binWidth'] = 25
        params['verbose'] = True
        extractor = featureextractor.RadiomicsFeatureExtractor(**params)
        extractor.enableAllImageTypes()
        extractor.enableAllFeatures()
    elif os.path.isfile(params):
        extractor = featureextractor.RadiomicsFeatureExtractor(params)
    else:
        extractor = featureextractor.RadiomicsFeatureExtractor(**params)
        extractor.enableAllImageTypes()
        extractor.enableAllFeatures()
    

    logger.info('Enabled input images types: %s', extractor.enabledImagetypes)
    logger.info('Enabled features: %s', extractor.enabledFeatures)
    logger.info('Current settings: %s', extractor.settings)

    headers = None
    amount = 0

    radiomics_array = []

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
            featureVector['Label'] = label
            
            try:
                featureVector.update(extractor.execute(imageFilepath, maskFilepath, 1))
                radiomics_array.append(dict(featureVector))
                amount = len(featureVector)
                logger.info("Extracted %d features", amount)
                logger.info("Features: %s", featureVector.keys())
                try:
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
                    logger.error('CSV WRITE FAILED', exc_info=True)
            except Exception:
                logger.error('FEATURE EXTRACTION FAILED', exc_info=True)

    return amount, radiomics_array