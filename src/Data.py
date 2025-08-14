from Imports import *
from Radiomics import get_batch_radiomics

def read_csv(csv_filepath):
    """
    Lee un archivo CSV y devuelve un DataFrame de pandas.
    """
    try:
        df = pd.read_csv(csv_filepath)
        amount_cols = len(df.columns)
        amount_rows = len(df.index)
        
        df = df.apply(pd.to_numeric, errors='ignore')

    except Exception as e:
        print(f"Error al leer el archivo CSV: {e}")
        return (0,0), None
    
    return (amount_rows, amount_cols), df

def split_by_fold(data, fold_column='FOLD', current_fold=1):
    """
    Divide los datos en entrenamiento y validación según el fold actual.

    Parámetros:
    - data: DataFrame con los datos completos.
    - fold_column: Nombre de la columna que contiene los números de los folds.
    - current_fold: Número del fold que se usará como conjunto de validación.

    Retorna:
    - train_data: Datos para entrenamiento.
    - val_data: Datos para validación.
    """
    val_data = data[data[fold_column] == current_fold]
    train_data = data[data[fold_column] != current_fold]

    return train_data.drop(columns=fold_column), val_data.drop(columns=fold_column)

def ingenio_dataset(base_path, 
                    output_csv):
    """
    Genera un archivo CSV con las rutas de las imágenes y segmentaciones en formato .nrrd.

    Parámetros:
    - base_path: Ruta base donde están las carpetas de los hospitales.
    - output_csv: Ruta del archivo CSV de salida.

    El CSV generado tendrá las columnas:
    - 'Hospital': Nombre del hospital.
    - 'Subject': Identificador del sujeto.
    - 'Image': Ruta de la imagen CT.
    - 'Mask': Ruta de la segmentación correspondiente.
    - 'Labels': Lista de etiquetas únicas encontradas en la segmentación.
    """

    data = []

    for hospital in os.listdir(base_path):
        hospital_path = os.path.join(base_path, hospital)
        if not os.path.isdir(hospital_path):
            continue

        for subject in os.listdir(hospital_path):
            subject_path = os.path.join(hospital_path, subject)
            if not os.path.isdir(subject_path):
                continue

            ct_files = [file for file in os.listdir(subject_path) if (file.endswith(".nrrd") and ("TH_CT" or "BODY_CT" in file) and (not file.endswith("_seg.nrrd")))]

            for ct_file in ct_files:
                image_path = os.path.join(subject_path, ct_file)
                segmentation_path = image_path.replace(".nrrd", "_seg.nrrd")
                unique_labels = []
                if os.path.exists(segmentation_path):

                    segmentation = sitk.ReadImage(segmentation_path)
                    segmentation_array = sitk.GetArrayFromImage(segmentation)
                    unique_labels = np.unique(segmentation_array)           

                    data.append({
                        "Hospital": hospital,
                        "Subject": subject,
                        "Image": image_path,
                        "Mask": segmentation_path,
                        "Labels": unique_labels
                    })
                         
    output_dir = os.path.dirname(output_csv)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["hospital", "subject", "Image", "Mask", "Labels"])
        writer.writeheader()
        writer.writerows(data)

    print(f"Archivo CSV generado en: {output_csv}")

def merge_with_labels(dataset_csv, labels_csv, output_csv):
    """
    Une el dataset con las columnas PFS_6m y OS_12m del archivo labels_csv.
    Guarda el resultado en un csv solo si CHECK es OK.
    Además, guarda una lista de sujetos del dataset que no están en df_labels.
    """

    df_dataset = pd.read_csv(dataset_csv)
    df_labels = pd.read_csv(labels_csv, delimiter=";")

    duplicados = df_dataset['subject'][df_dataset['subject'].duplicated()]
    if not duplicados.empty:
        print("Sujetos repetidos encontrados en df_dataset:")
        print(duplicados)
    else:
        print("No hay sujetos repetidos en df_dataset.")

    df_labels['case_id'] = df_labels['case_id'].str.replace('-', '_')
    df_labels = df_labels[['case_id', 'PFS_6m', 'OS_12m', 'CHECK']]

    merged = df_dataset.merge(df_labels, left_on='subject', right_on='case_id', how='left')

    columnas = ["hospital", "subject", "Image", "Mask", "Labels", "PFS_6m", "OS_12m"]
    merged = merged[columnas + ['CHECK']]

    merged = merged.fillna('')

    merged = merged[merged['CHECK'] == 'OK']

    for col in ["PFS_6m", "OS_12m"]:
        merged[col] = merged[col].apply(lambda x: int(float(x)) if str(x).strip() != '' else '' )

    merged = merged[columnas]

    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columnas)
        writer.writeheader()
        writer.writerows(merged.to_dict(orient='records'))
    print(f"Archivo CSV combinado guardado en: {output_csv}")

    invalid_labels = set(
        merged[
            (~merged['PFS_6m'].isin([0, 1])) & (~merged['OS_12m'].isin([0, 1]))
        ]['subject'].unique()
    )

    if invalid_labels:
        txt_path = output_csv.replace('.csv', '_invalid_labels.txt')
        with open(txt_path, 'w') as f:
            for subj in sorted(invalid_labels):
                f.write(f"{subj}\n")
        print(f"Lista de sujetos con etiquetas inválidas guardada en: {txt_path}")
    else:
        print("Todos los sujetos tienen valores válidos en PFS_6m u OS_12m.")

def save_stratified_folds(input_csv):
    """
    Realiza un StratifiedKFold de 5 y 10 folds para PFS_6m y OS_12m y guarda los resultados en cuatro archivos Excel.
    """

    df = pd.read_csv(input_csv)
    print(f"Total registros en el dataset: {len(df)}")
    print(f"Columnas del dataset: {df.columns.tolist()}")

    for label in ["PFS_6m", "OS_12m"]:

        df_label = df[df[label].notna() & (df[label] != '')].copy()
        df_label[label] = df_label[label].astype(int)

        for n_splits in [5, 10]:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            df_folds = df_label.copy()
            df_folds["FOLD"] = -1

            for fold, (_, val_idx) in enumerate(skf.split(df_folds, df_folds[label])):
                df_folds.iloc[val_idx, df_folds.columns.get_loc("FOLD")] = fold

            output_csv = f".\\data\\ingenio_split_labels_{label}_{n_splits}folds.csv"
            fieldnames = ["hospital", "subject", "Image", "Mask", "Labels", label, "FOLD"]

            with open(output_csv, mode='w', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(df_folds[fieldnames].to_dict(orient='records'))
            print(f"Archivo CSV combinado guardado en: {output_csv}")

def merge_labels_with_radiomics(input_csv, radiomics_csv, output_csv, label_cols=['PFS_6m', 'OS_12m']):
    """
    Une el dataset de radiomics con las columnas PFS_6m y OS_12m del archivo labels_csv.
    Guarda el resultado en un csv.
    """

    df_radiomics = pd.read_csv(radiomics_csv, delimiter=",", quotechar='"')
    df_labels = pd.read_csv(input_csv)

    print(df_radiomics.columns)

    if isinstance(df_radiomics.index, pd.MultiIndex):
        df_radiomics = df_radiomics.reset_index()


    expected_names = ['hospital', 'subject', 'Image', 'Mask']
    current_names = list(df_radiomics.columns[:4])
    if all(str(col).startswith('level_') or str(col) == '' for col in current_names):
        rename_dict = {df_radiomics.columns[i]: expected_names[i] for i in range(4)}
        df_radiomics = df_radiomics.rename(columns=rename_dict)

    df_labels['subject'] = df_labels['subject'].astype(str)#.str.strip()
    df_radiomics['subject'] = df_radiomics['subject'].astype(str)#.str.strip()

    merged = pd.merge(df_radiomics, df_labels, on='subject', how='left', suffixes=('', '_radiomics'))

    print(merged.head(10))

    for col in label_cols:
        if col in merged.columns:
            merged[col] = merged[col].apply(lambda x: int(float(x)) if str(x).strip() != '' and not pd.isna(x) else '')
        else:
            print(f"Warning: La columna '{col}' no existe en el DataFrame y no será procesada.")
    
    print(f"Total registros en el dataset combinado: {len(merged)}")

    csv_OS_12m = output_csv.replace('.csv', '_OS_12m.csv')
    merged_OS_12m = merged[merged['OS_12m_radiomics'].notna() & (merged['OS_12m_radiomics'] != '')].copy()
    print(f"Total registros con OS_12m: {len(merged_OS_12m)}")
    merged_OS_12m['Label'] = merged_OS_12m['OS_12m_radiomics'].astype(int)
    cols_to_drop_os = [col for col in merged_OS_12m.columns if col.endswith('_radiomics')] + ['PFS_6m', 'OS_12m']
    merged_OS_12m.drop(columns=cols_to_drop_os, inplace=True, errors='ignore')

    csv_PFS_6m = output_csv.replace('.csv', '_PFS_6m.csv')
    merged_PFS_6m = merged[merged['PFS_6m_radiomics'].notna() & (merged['PFS_6m_radiomics'] != '')].copy()
    print(f"Total registros con PFS_6m: {len(merged_PFS_6m)}")
    merged_PFS_6m['Label'] = merged_PFS_6m['PFS_6m_radiomics'].astype(int)
    cols_to_drop_pfs = [col for col in merged_PFS_6m.columns if col.endswith('_radiomics')] + ['PFS_6m', 'OS_12m']
    merged_PFS_6m.drop(columns=cols_to_drop_pfs, inplace=True, errors='ignore')

    merged.to_csv(output_csv, index=False)
    merged_OS_12m.to_csv(csv_OS_12m, index=False)
    merged_PFS_6m.to_csv(csv_PFS_6m, index=False)
    print(f"Archivo CSV combinado guardado en: {output_csv}")