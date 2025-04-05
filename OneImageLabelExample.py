import numpy as np
import pandas as pd
import pydicom
import os


patient_details_df=pd.read_csv('/projectnb/dl4ds/students/atuladas/DS542-CNN-A/Project-Datase.csv')
print(len(patient_details_df))
dicom_path='/projectnb/dl4ds/students/atuladas/DS542-CNN-A/Midline Train Test/train/006_S_4192_2011-09-27_Slice_86.dcm'
file_path=os.path.basename(dicom_path)
subject_id='_'.join(file_path.split('_')[:3])
print(subject_id)


matching_row=patient_details_df[patient_details_df['Subject']==subject_id]
print(matching_row)
if not matching_row.empty:
    label_info = matching_row.iloc[0].to_dict()
    print(label_info)
    print(f"✅ Match found for subject {subject_id}:\n")
    print("Subject:",label_info["Subject"],"\n")
    print("Diagnosis:",label_info["Group"],"\n")
else:
    print(f"❌ No match found for subject {subject_id}")
