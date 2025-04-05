import pydicom
import pandas as pd

# Step 1: Load the DICOM file
dcm_path = "Midline Train Test/train/002_S_0685_2011-07-08_Slice_86.dcm"
dcm = pydicom.dcmread(dcm_path)
patient_id = dcm.PatientID  # e.g., '002_S_0685'

# Step 2: Load the CSV
csv_path = "Project-Datase.csv"
df = pd.read_csv(csv_path)

# Step 3: Look up the diagnosis using the Subject column
matching_row = df[df['Subject'] == patient_id]

if not matching_row.empty:
    diagnosis = matching_row.iloc[0]['Group']  # e.g., 'AD'
    print(f"Subject: {patient_id}")
    print(f"Diagnosis: {diagnosis}")
else:
    print(f"No matching record found for Subject: {patient_id}")
