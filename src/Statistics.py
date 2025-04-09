from Imports import *
from scipy.stats import shapiro, levene

def pearson_correlation(features, outPath = ''):
    """
    Filtra si coeficiente de correlación de Pearson ≤ 0.95.
    """
    if isinstance(features, list) or isinstance(features, np.ndarray):
        features = pd.DataFrame(features)
    
    exclude_columns = [col for col in features.columns if col in ['Image', 'Mask', 'Label'] or col.startswith('diagnostics_')]

    features_excluded = features.drop(columns=exclude_columns, errors='ignore')

    correlation_matrix = features_excluded.corr().abs()
    print("Matriz de correlación de Pearson:\n", correlation_matrix)

    to_drop = set()
    for i in range(correlation_matrix.shape[0]):
        for j in range(i):
            if correlation_matrix.iloc[i, j] > 0.95:
                to_drop.add(correlation_matrix.columns[i])
                with open(outPath, "a") as log_file:
                    log_file.write(f"Feature '{correlation_matrix.columns[i]}' está correlacionada con '{correlation_matrix.columns[j]}' (coeficiente: {correlation_matrix.iloc[i, j]:.2f})\n")

    initial_feature_count = features_excluded.shape[1]
    final_feature_count = initial_feature_count - len(to_drop)
    with open(outPath, "a") as log_file:
        log_file.write(f"\n\nCantidad inicial de características: {initial_feature_count}\n")
        log_file.write(f"Cantidad final de características: {final_feature_count}\n")
    filtered_features = features.drop(columns=to_drop)

    return filtered_features

def anova_ftest(features, outPath='', k=10):
    """
    Filtra las K características principales según el valor F de ANOVA.
    """
    if isinstance(features, list) or isinstance(features, np.ndarray):
        features = pd.DataFrame(features)

    exclude_columns = [col for col in features.columns if col in ['Image', 'Mask', 'Label'] or col.startswith('diagnostics_')]

    filtered_features = features.drop(columns=exclude_columns, errors='ignore')

    # Inicializa un DataFrame para almacenar los resultados
    results = pd.DataFrame(columns=['Feature', 'F-statistic', 'p-value'])

    rows = []
    # Realiza la prueba ANOVA para cada característica
    for column in filtered_features.columns:
        groups = [filtered_features[column][np.array(features['Label']) == label] for label in np.unique(features['Label'])]
        if check_anova_assumptions(groups) == False:
            continue
        f_statistic, p_value = f_oneway(*groups)
        rows.append({'Feature': column, 'F-statistic': f_statistic, 'p-value': p_value})
    results = pd.concat([results, pd.DataFrame(rows)], ignore_index=True)
    # Ordena las características por el valor F en orden descendente
    results = results.sort_values(by='F-statistic', ascending=False)

    # Selecciona las K características principales
    top_features = results.head(k)['Feature'].tolist()
    
    with open(outPath, "a") as log_file:
        log_file.write(f"\n\nCantidad inicial de características: {len(filtered_features.columns)}\n")
        log_file.write(f"Cantidad final de características: {len(top_features)}\n")
        log_file.write("Características seleccionadas:\n")
        for feature in top_features:
            f_statistic = results.loc[results['Feature'] == feature, 'F-statistic'].values[0]
            log_file.write(f"{feature}: F-statistic = {f_statistic:.2f}\n")
    
    features_all = top_features + exclude_columns

    return features[features_all]

def check_anova_assumptions(groups):
    """
    Verifica las suposiciones de ANOVA:
    - Normalidad: Prueba de Shapiro-Wilk.
    - Homogeneidad de varianza: Prueba de Levene.
    """
    # Normalidad: Shapiro-Wilk test
    for group in groups:
        if len(group) < 3:  # Shapiro-Wilk requires at least 3 samples
            return False
        stat, p_value = shapiro(group)
        if p_value < 0.05:  # Reject null hypothesis of normality
            return False

    # Homogeneidad de varianza: Levene's test
    stat, p_value = levene(*groups)
    if p_value < 0.05:  # Reject null hypothesis of equal variances
        return False

    return True