from Imports import *
from Radiomics import get_batch_radiomics

import os
import csv

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

def ingenio_dataset(base_path="\\\\10.5.38.120\\BIT-UPM-projects\\INGENIO-RAD\\DATA\\cleanData\\NRRD", output_csv="..\\data\\ingenio_dataset.csv"):
    """
    TODO: quiero especificar qué labels hay en la segmentación

    Genera un archivo CSV con las rutas de las imágenes y segmentaciones en formato .nrrd.

    Parámetros:
    - base_path: Ruta base donde están las carpetas de los hospitales.
    - output_csv: Ruta del archivo CSV de salida.

    El CSV generado tendrá las columnas:
    - 'hospital': Nombre del hospital.
    - 'subject': Identificador del sujeto.
    - 'image_path': Ruta de la imagen CT.
    - 'segmentation_path': Ruta de la segmentación correspondiente.
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
                        "hospital": hospital,
                        "subject": subject,
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



ingenio_dataset()
get_batch_radiomics(inputCSV="..\\data\\ingenio_dataset.csv",
                    outPath="..\\data\\ingenio_radiomics.csv", 
                    progress_filename="..\\data\\radiomics_log.txt",
                    params="..\\data\\Params.yaml")