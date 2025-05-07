from Imports import *



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





