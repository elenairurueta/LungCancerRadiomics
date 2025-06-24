import os
import json
import pandas as pd

rows = []

path = "//10.5.38.120/BIT-UPM-projects/NODULES/SCRATCH_STUDENTS/FJD_Screening/markups/benign"

for patient in os.listdir(path):
    patient_path = os.path.join(path, patient)
    if os.path.isdir(patient_path):
        for file in os.listdir(patient_path):
            if not file.endswith('.mrk.json'):
                raise ValueError(f"File {file} in patient {patient} does not end with .mrk.json")
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

path = "//10.5.38.120/BIT-UPM-projects/NODULES/SCRATCH_STUDENTS/FJD_Screening/markups/malignant"

for patient in os.listdir(path):
    patient_path = os.path.join(path, patient)
    if os.path.isdir(patient_path):
        for file in os.listdir(patient_path):
            if not file.endswith('.mrk.json'):
                raise ValueError(f"File {file} in patient {patient} does not end with .mrk.json")
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



