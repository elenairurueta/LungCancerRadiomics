import os
import json
import pandas as pd

def nodules_locations():
    """
    Extrae las localizaciones de los nódulos desde archivos JSON en los directorios especificados y los guarda en un archivo Excel.
    Procesa nódulos benignos y malignos por separado, extrayendo paciente, TAC, etiqueta del nódulo y posición.
    """
    rows = []
    #BENIGNOS
    path = "//10.5.38.120/BIT-UPM-projects/NODULES/SCRATCH_STUDENTS/FJD_Screening/markups/benign"
    for patient in os.listdir(path):
        patient_path = os.path.join(path, patient)
        if os.path.isdir(patient_path):
            for file in os.listdir(patient_path):
                if not file.endswith('.mrk.json'):
                    raise ValueError(f"El archivo {file} en el paciente {patient} no termina en .mrk.json")
                file_path = os.path.join(patient_path, file)
                file_name = file.replace('.mrk.json', '')
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for markup in data.get("markups", []):
                        for cp in markup.get("controlPoints", []):
                            label = (cp.get("label")).replace("-", "_")
                            position = cp.get("position")
                            rows.append({
                                "Patient": patient,
                                "CT": file_name,
                                "Nodule": label,
                                "Position_X": position[0],
                                "Position_Y": position[1],
                                "Position_Z": position[2],
                                'Label': 0
                            })
    #MALIGNOS
    path = "//10.5.38.120/BIT-UPM-projects/NODULES/SCRATCH_STUDENTS/FJD_Screening/markups/malignant"
    for patient in os.listdir(path):
        patient_path = os.path.join(path, patient)
        if os.path.isdir(patient_path):
            for file in os.listdir(patient_path):
                if not file.endswith('.mrk.json'):
                    raise ValueError(f"El archivo {file} en el paciente {patient} no termina en .mrk.json")
                file_path = os.path.join(patient_path, file)
                file_name = file.replace('.mrk.json', '')
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for markup in data.get("markups", []):
                        for cp in markup.get("controlPoints", []):
                            label = (cp.get("label")).replace("-", "_")
                            position = cp.get("position")
                            rows.append({
                                "Patient": patient,
                                "CT": file_name,
                                "Nodule": label,
                                "Position_X": position[0],
                                "Position_Y": position[1],
                                "Position_Z": position[2],
                                'Label': 1
                            })

    df = pd.DataFrame(rows)
    df.sort_values(by=["Patient", "CT", "Nodule"], inplace=True)
    df.to_excel("FJD_Screening_Locations.xlsx", index=False)

def estadisticas_tac():
    """
    Calcula estadísticas de TAC, nódulos benignos y malignos por paciente.
    """
    excel_path = ".\\data\\FJD_Screening.xlsx"
    pacientes = pd.read_excel(excel_path, sheet_name='Pacientes')
    tacs = pd.read_excel(excel_path, sheet_name='TAC')
    nodulos = pd.read_excel(excel_path, sheet_name='Nodulos')
    canceres = pd.read_excel(excel_path, sheet_name='Canceres')

    # TAC por paciente
    tac_por_paciente = tacs.groupby('ID_Paciente').size()
    print(f"Promedio de TAC por paciente: {tac_por_paciente.mean():.2f} ± {tac_por_paciente.std():.2f}")

    pacientes_cancer = pacientes[pacientes['Cancer'] == 1]['ID_Paciente']
    pacientes_no_cancer = pacientes[pacientes['Cancer'] == 0]['ID_Paciente']

    tac_cancer = tacs[tacs['ID_Paciente'].isin(pacientes_cancer)].groupby('ID_Paciente').size()
    tac_no_cancer = tacs[tacs['ID_Paciente'].isin(pacientes_no_cancer)].groupby('ID_Paciente').size()

    print(f"Promedio de TAC por paciente con cáncer: {tac_cancer.mean():.2f} ± {tac_cancer.std():.2f}")
    print(f"Promedio de TAC por paciente sin cáncer: {tac_no_cancer.mean():.2f} ± {tac_no_cancer.std():.2f}")

    # Nódulos benignos por TAC (incluyendo TAC sin nódulos)
    nodulos_por_tac = tacs[['ID_TAC']].merge(
        nodulos.groupby('ID_TAC').size().rename('n_nodulos'),
        left_on='ID_TAC', right_index=True, how='left'
    ).fillna(0)
    print(f"Promedio de nódulos benignos por TAC: {nodulos_por_tac['n_nodulos'].mean():.2f} ± {nodulos_por_tac['n_nodulos'].std():.2f}")

    # Nódulos malignos por TAC (incluyendo TAC sin nódulos malignos)
    canceres_por_tac = tacs[['ID_TAC']].merge(
        canceres.groupby('ID_TAC').size().rename('n_canceres'),
        left_on='ID_TAC', right_index=True, how='left'
    ).fillna(0)
    print(f"Promedio de nódulos malignos por TAC: {canceres_por_tac['n_canceres'].mean():.2f} ± {canceres_por_tac['n_canceres'].std():.2f}")

    # Nódulos por paciente
    nodulos_tac = nodulos.merge(tacs[['ID_TAC', 'ID_Paciente']], on='ID_TAC', how='left')
    nodulos_por_paciente = nodulos_tac.groupby('ID_Paciente').size()
    print(f"Promedio de nódulos benignos por paciente: {nodulos_por_paciente.mean():.2f} ± {nodulos_por_paciente.std():.2f}")

    # Nódulos malignos por paciente
    canceres_tac = canceres.merge(tacs[['ID_TAC', 'ID_Paciente']], on='ID_TAC', how='left')
    canceres_por_paciente = canceres_tac.groupby('ID_Paciente').size()
    print(f"Promedio de nódulos malignos por paciente: {canceres_por_paciente.mean():.2f} ± {canceres_por_paciente.std():.2f}")

    # Años de seguimiento por paciente (diferencia entre primer y último TAC)
    tacs['Fecha_TAC'] = pd.to_datetime(tacs['Fecha_TAC'], errors='coerce')

    seguimiento = tacs.groupby('ID_Paciente')['Fecha_TAC'].agg(['min', 'max'])
    seguimiento['years'] = (seguimiento['max'] - seguimiento['min']).dt.days / 365.25

    # Pacientes con cáncer
    years_cancer = seguimiento.loc[seguimiento.index.isin(pacientes_cancer), 'years']
    print(f"Años de seguimiento por paciente con cáncer: {years_cancer.mean():.2f} ± {years_cancer.std():.2f}")

    # Pacientes sin cáncer
    years_no_cancer = seguimiento.loc[seguimiento.index.isin(pacientes_no_cancer), 'years']
    print(f"Años de seguimiento por paciente sin cáncer: {years_no_cancer.mean():.2f} ± {years_no_cancer.std():.2f}")

    # Cantidad total de TAC
    total_tac = tacs.shape[0]
    print(f"Cantidad total de TAC: {total_tac}")

    # Cantidad de TAC en pacientes con cáncer
    tac_con_cancer = tacs[tacs['ID_Paciente'].isin(pacientes_cancer)].shape[0]
    print(f"Cantidad de TAC en pacientes con cáncer: {tac_con_cancer}")

    # Cantidad de TAC en pacientes sin cáncer
    tac_sin_cancer = tacs[tacs['ID_Paciente'].isin(pacientes_no_cancer)].shape[0]
    print(f"Cantidad de TAC en pacientes sin cáncer: {tac_sin_cancer}")

    # Agregar columna de número de TAC por paciente
    num_tac = tacs.groupby('ID_Paciente').size().rename('Num_TAC')
    pacientes = pacientes.merge(num_tac, left_on='ID_Paciente', right_index=True, how='left').fillna({'Num_TAC': 0})

    # Agregar columna de número de nódulos por TAC
    num_nodulos = nodulos.groupby('ID_TAC').size().rename('Num_Nodulos')
    tacs = tacs.merge(num_nodulos, left_on='ID_TAC', right_index=True, how='left').fillna({'Num_Nodulos': 0})

    # Guardar los cambios en el Excel (sobrescribe el archivo)
    with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        pacientes.to_excel(writer, sheet_name='Pacientes', index=False)
        tacs.to_excel(writer, sheet_name='TAC', index=False)
        nodulos.to_excel(writer, sheet_name='Nodulos', index=False)
        canceres.to_excel(writer, sheet_name='Canceres', index=False)

def check_folder_files(root_folder, extension):
    """
    Revisa si en cada subcarpeta de root_folder todos los archivos tienen la extensión indicada.
    Imprime advertencias en español si hay archivos no válidos o subcarpetas vacías.
    """
    all_ok = True
    for subdir, dirs, files in os.walk(root_folder):
        if subdir == root_folder:
            continue
        file_list = [f for f in files if not f.startswith('.')]
        if not file_list:
            print(f"Subcarpeta vacía: {subdir}")
            all_ok = False
        for f in file_list:
            if not f.lower().endswith(extension):
                print(f"Archivo no válido en {subdir}: {f}")
                all_ok = False
    if all_ok:
        print(f"Todos los archivos en {root_folder} y subcarpetas son {extension}")
    else:
        print(f"Hay archivos no {extension} en {root_folder}")
