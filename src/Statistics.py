from Imports import *

def pearson_correlation(features, outPath = ''):
    """
    Filtra si el coeficiente de correlación de Pearson ≤ 0.95.
    """
    if isinstance(features, list) or isinstance(features, np.ndarray):
        features = pd.DataFrame(features)
    
    exclude_columns = [col for col in features.columns if col in ['Image', 'Mask', 'Label', 'SeriesInstanceUID', 'AnnotationID', 'label'] or col.startswith('diagnostics_')]

    features_excluded = features.drop(columns=exclude_columns, errors='ignore')

    correlation_matrix = features_excluded.corr().abs()
    print("Matriz de correlación de Pearson:\n", correlation_matrix)

    to_drop = set()
    for i in range(correlation_matrix.shape[0]):
        for j in range(i):
            if correlation_matrix.iloc[i, j] > 0.95:
                to_drop.add(correlation_matrix.columns[i])
                with open(outPath, "a") as log_file:
                    log_file.write(f"La característica '{correlation_matrix.columns[i]}' está correlacionada con '{correlation_matrix.columns[j]}' (coeficiente: {correlation_matrix.iloc[i, j]:.2f})\n")
    # TODO: no se está repitiendo porque la matriz es simétrica?
    initial_feature_count = features_excluded.shape[1]
    final_feature_count = initial_feature_count - len(to_drop)

    with open(outPath, "a") as log_file:
        log_file.write(f"\n\nCantidad inicial de características: {initial_feature_count}\n")
        log_file.write(f"Cantidad final de características: {final_feature_count}\n")
    filtered_features = features.drop(columns=to_drop)

    return filtered_features

def anova_feature_selection(features, k=10, outPath=''):
    """
    Filtra las K características principales según el valor F de ANOVA.
    """


    if isinstance(features, list) or isinstance(features, np.ndarray):
        features = pd.DataFrame(features)

    exclude_columns = [col for col in features.columns if col in ['Image', 'Mask', 'Label', 'SeriesInstanceUID', 'AnnotationID', 'label'] or col.startswith('diagnostics_')]

    filtered_features = features.drop(columns=exclude_columns, errors='ignore')

    results = pd.DataFrame(columns=['Feature', 'F-statistic', 'p-value'])

    rows = []
    for column in filtered_features.columns:
        groups = [filtered_features[column][np.array(features['Label' if 'Label' in features.columns else 'label']) == label] for label in np.unique(features['Label' if 'Label' in features.columns else 'label'])]
        # if check_anova_assumptions(groups) == False:
        #     continue
        f_statistic, p_value = f_oneway(*groups)
        rows.append({'Feature': column, 'F-statistic': f_statistic, 'p-value': p_value})
    results = pd.concat([results, pd.DataFrame(rows)], ignore_index=True)
    results = results.sort_values(by='F-statistic', ascending=False)

    top_features = results.head(k)['Feature'].tolist()
    
    with open(outPath, "a") as log_file:
        log_file.write(f"\n\nCaracterísticas seleccionadas por ANOVA con k = {k}:\n")
        log_file.write(f"\n\nCantidad inicial de características: {len(filtered_features.columns)}\n")
        log_file.write(f"Cantidad final de características: {len(top_features)}\n")
        log_file.write("Características seleccionadas:\n")
        for feature in top_features:
            f_statistic = results.loc[results['Feature'] == feature, 'F-statistic'].values[0]
            log_file.write(f"{feature}: F-statistic = {f_statistic:.2f}\n")
    
    features_all = top_features + exclude_columns
    print(top_features)
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

def clustering_feature_selection(features, k=10, outPath=''):
    """
    Reduce la dimensionalidad de los datos agrupando características similares en clusters jerárquicos.
    
    Parámetros:
    - features: DataFrame de características.
    - k: Número deseado de clusters.
    - outPath: Ruta para guardar el log de las características agrupadas.

    Retorna:
    - DataFrame con una característica representativa por cluster.
    """
    if isinstance(features, list) or isinstance(features, np.ndarray):
        features = pd.DataFrame(features)

    exclude_columns = [col for col in features.columns if col in ['Image', 'Mask', 'Label', 'label', 'SeriesInstanceUID', 'AnnotationID'] or col.startswith('diagnostics_')]
    filtered_features = features.drop(columns=exclude_columns, errors='ignore')

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(filtered_features)

    linkage_matrix = linkage(scaled_features.T, method='ward')
    cluster_labels = fcluster(linkage_matrix, k, criterion='maxclust') 

    cluster_map = pd.DataFrame({'Feature': filtered_features.columns, 'Cluster': cluster_labels})

    representative_features = []
    for cluster in range(1, k + 1):
        cluster_features = cluster_map[cluster_map['Cluster'] == cluster]['Feature']
        representative_features.append(cluster_features.iloc[0])

    with open(outPath, "a") as log_file:
        log_file.write(f"\n\nCaracterísticas agrupadas por clusters con k = {k}:\n")
        log_file.write(f"\n\nCantidad inicial de características: {len(filtered_features.columns)}\n")
        log_file.write(f"Cantidad final de características (clusters): {len(representative_features)}\n")
        log_file.write("Características representativas por cluster:\n")
        for feature in representative_features:
            log_file.write(f"{feature}\n")

    features_all = representative_features + exclude_columns

    return features[features_all]

def sfm_feature_selection(features, k=10, outPath=''):
    """
    Selecciona las K características principales basándose en la importancia de características usando un modelo Random Forest.
    
    Parámetros:
    - features: DataFrame de características.
    - k: Número de características principales a seleccionar.
    - outPath: Ruta para guardar el log de las características seleccionadas.

    Retorna:
    - DataFrame con las K características seleccionadas.
    """
    if isinstance(features, list) or isinstance(features, np.ndarray):
        features = pd.DataFrame(features)

    exclude_columns = [col for col in features.columns if col in ['Image', 'Mask', 'Label', 'label', 'SeriesInstanceUID', 'AnnotationID'] or col.startswith('diagnostics_')]
    
    filtered_features = features.drop(columns=exclude_columns, errors='ignore')
    labels = features['Label' if 'Label' in features.columns else 'label']
    rf = RandomForestClassifier(random_state=42)
    rf.fit(filtered_features, labels)

    sfm = SelectFromModel(rf, prefit=True, max_features=k)
    selected_features_mask = sfm.get_support()
    selected_features = filtered_features.columns[selected_features_mask]

    with open(outPath, "a") as log_file:
        log_file.write(f"\n\nCaracterísticas seleccionadas por SFM con k = {k}:\n")
        log_file.write(f"\n\nCantidad inicial de características: {len(filtered_features.columns)}\n")
        log_file.write(f"Cantidad final de características: {len(selected_features)}\n")
        log_file.write("Características seleccionadas:\n")
        for feature in selected_features:
            log_file.write(f"{feature}\n")

    features_all = list(selected_features) + exclude_columns

    return features[features_all]

def pca_feature_selection(features, k=10, outPath=''):
    """
    Reduce la dimensionalidad de los datos utilizando PCA seleccionando los K componentes principales.
    
    Parámetros:
    - features: DataFrame de características.
    - k: Número de componentes principales a retener.
    - outPath: Ruta para guardar el log de los componentes seleccionados.

    Retorna:
    - DataFrame con los K componentes principales.
    """
    if isinstance(features, list) or isinstance(features, np.ndarray):
        features = pd.DataFrame(features)

    exclude_columns = [col for col in features.columns if col in ['Image', 'Mask', 'Label', 'label', 'SeriesInstanceUID', 'AnnotationID'] or col.startswith('diagnostics_')]
    filtered_features = features.drop(columns=exclude_columns, errors='ignore')

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(filtered_features)

    pca = PCA(n_components=k)
    principal_components = pca.fit_transform(scaled_features)

    pca_columns = [f"PC{i+1}" for i in range(k)]
    pca_df = pd.DataFrame(principal_components, columns=pca_columns, index=features.index)

    explained_variance = pca.explained_variance_ratio_
    with open(outPath, "a") as log_file:
        log_file.write(f"\n\nSelección de características por PCA con k = {k}:\n")
        log_file.write(f"Varianza explicada por cada componente:\n")
        for i, variance in enumerate(explained_variance):
            log_file.write(f"PC{i+1}: {variance:.4f}\n")
        log_file.write(f"Varianza total explicada: {sum(explained_variance):.4f}\n")

    return pd.concat([pca_df, features[exclude_columns]], axis=1)

def shap_feature_selection(features, k=10, outPath=''):
    return