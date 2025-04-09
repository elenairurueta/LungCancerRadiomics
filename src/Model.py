from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from Imports import *

class StreamToLogger:
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.line_buffer = ''

    def write(self, message):
        if message.strip():  # Evitar líneas vacías
            self.logger.log(self.log_level, message.strip())

    def flush(self):  # Método requerido para compatibilidad
        pass

def model_random_forest(features, outPath_model='', outPath_log='', n_trees=100, max_depth=3, min_samples_split=0.5, crossVal=True):

    exclude_columns = [col for col in features.columns if col in ['Image', 'Mask', 'Label'] or col.startswith('diagnostics_')]
    filtered_features = features.drop(columns=exclude_columns, errors='ignore')
    
    X = filtered_features.to_numpy()
    y = features['Label'].to_numpy()

    logging.basicConfig(filename=outPath_log, level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()
    original_stdout = sys.stdout
    
    try:
        sys.stdout = StreamToLogger(logger, logging.INFO)
        rf = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth, min_samples_split=min_samples_split, 
                                    oob_score=True, random_state=0, verbose=3)
        
        if(crossVal):
            scoring = ['precision_macro', 'recall_macro', 'accuracy', 'roc_auc']
            scores = cross_validate(rf, X, y, cv=4, scoring=scoring)
            logger.info(f"Fit time: {scores['fit_time'].mean()}")
            logger.info(f"Score time: {scores['score_time'].mean()}")
            for score in scores:
                if score not in ('fit_time', 'score_time'):
                    logger.info(f"{score}: " + "(%0.2f +- %0.2f)" % (scores[score].mean(), scores[score].std()))
        else:
            rf.fit(X, y)
            os.makedirs(os.path.dirname(outPath_model), exist_ok=True)
            scores = []
        dump(rf, outPath_model) 
    finally:
        sys.stdout = original_stdout

    return scores



