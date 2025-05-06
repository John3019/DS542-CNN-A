import os
import pydicom
import pandas as pd
from collections import Counter

# Load the project dataset
csv_path = "Project-Datase.csv"
df = pd.read_csv(csv_path)

# Path to the train folder
train_folder = "Midline Train Test/train"

# Diagnosis count tracker
diagnosis_counter = Counter()

# Loop through all DICOM files in the train folder
for root, _, files in os.walk(train_folder):
    for file in files:
        if file.endswith('.dcm'):
            dcm_path = os.path.join(root, file)
            try:
                dcm = pydicom.dcmread(dcm_path)
                patient_id = dcm.PatientID
                match = df[df['Subject'] == patient_id]
                if not match.empty:
                    diagnosis = match.iloc[0]['Group']
                    diagnosis_counter[diagnosis] += 1
                else:
                    diagnosis_counter['Unknown'] += 1
            except Exception as e:
                print(f"Error reading {dcm_path}: {e}")
                diagnosis_counter['Error'] += 1

# Print final counts
print("Diagnosis Counts in TRAIN folder:")
for diag, count in diagnosis_counter.items():
    print(f"{diag}: {count}")
