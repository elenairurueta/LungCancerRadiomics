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

def ingenio_dataset(base_path="\\\\10.5.38.120\\BIT-UPM-projects\\INGENIO-RAD\\DATA\\cleanData\\NRRD", 
                    output_csv=".\\data\\ingenio_dataset.csv"):
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

def merge_with_labels(dataset_csv=".\\data\\ingenio_dataset_cleanBASAL.csv", 
                      labels_csv="\\\\10.5.38.120\\BIT-UPM-projects\\INGENIO-RAD\\DATA\\csv\\output\\INGENIO_output_2025-05-21.csv", 
                      output_csv=".\\data\\ingenio_dataset_cleanBASAL_labels.csv"):
    """
    Une el dataset con las columnas PFS_6m y OS_12m del archivo labels_csv.
    Guarda el resultado en un csv.
    """

    df_dataset = pd.read_csv(dataset_csv)
    df_labels = pd.read_csv(labels_csv)

    df_labels['record_id'] = df_labels['record_id'].str.replace('-', '_')
    df_labels = df_labels[['record_id', 'PFS_6m', 'OS_12m']]

    merged = df_dataset.merge(df_labels, left_on='subject', right_on='record_id', how='left')

    columnas = ["hospital", "subject", "Image", "Mask", "Labels", "PFS_6m", "OS_12m"]
    merged = merged[columnas]

    merged = merged.fillna('')

    for col in ["PFS_6m", "OS_12m"]:
        merged[col] = merged[col].apply(lambda x: int(float(x)) if str(x).strip() != '' else '')

    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columnas)
        writer.writeheader()
        writer.writerows(merged.to_dict(orient='records'))
    print(f"Archivo CSV combinado guardado en: {output_csv}")

def save_stratified_folds(input_csv=".\\data\\ingenio_dataset_cleanBASAL_labels.csv"):
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



# ingenio_dataset()
# merge_with_labels()
# save_stratified_folds(input_csv="C:\\dev\\LungCancerRadiomics\\data\\ingenio_dataset_cleanBASAL_labels.csv")
# get_batch_radiomics(inputCSV="C:\\dev\\LungCancerRadiomics\\data\\ingenio_dataset_cleanBASAL.csv",
#                     outPath="C:\\dev\\LungCancerRadiomics\\data\\ingenio_radiomics_cleanBASAL.csv", 
#                     progress_filename="C:\\dev\\LungCancerRadiomics\\data\\radiomics_log.txt",
#                     params="C:\\dev\\LungCancerRadiomics\\data\\Params.yaml")
