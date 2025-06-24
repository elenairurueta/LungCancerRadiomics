import pandas as pd
import os

excel_path = r'.\data\FJD_Screening.xlsx'
pacientes = pd.read_excel(excel_path, sheet_name='Pacientes')
tacs = pd.read_excel(excel_path, sheet_name='TAC')
nodulos = pd.read_excel(excel_path, sheet_name='Nodulos')
canceres = pd.read_excel(excel_path, sheet_name='Canceres')

#TAC por paciente
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

# def check_folder_files(root_folder, extension):
#     all_ok = True
#     for subdir, dirs, files in os.walk(root_folder):
#         if subdir == root_folder:
#             continue  # skip root itself
#         file_list = [f for f in files if not f.startswith('.')]
#         if not file_list:
#             print(f"Subcarpeta vacía: {subdir}")
#             continue
#         for f in file_list:
#             if not f.lower().endswith(extension):
#                 print(f"Archivo no válido en {subdir}: {f}")
#                 all_ok = False
#     if all_ok:
#         print(f"Todos los archivos en {root_folder} y subcarpetas son {extension}")
#     else:
#         print(f"Hay archivos no {extension} en {root_folder}")

# path = "//10.5.38.120/BIT-UPM-projects/NODULES/SCRATCH_STUDENTS/FJD_Screening/markups/malignant"
# # Cambia los paths según tu estructura
# check_folder_files(path + r'/NRRD', '.nrrd')
# check_folder_files(path + r'/markups', '.mrk.json')
