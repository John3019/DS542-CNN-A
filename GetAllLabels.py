import pandas as pd
import numpy as np
import pydicom
import os

files_path='/projectnb/dl4ds/students/atuladas/DS542-CNN-A/Midline Train Test/train'
patient_details_df=pd.read_csv('/projectnb/dl4ds/students/atuladas/DS542-CNN-A/Project-Datase.csv')


file_path=os.listdir(files_path)
#print(file_path)
file_names=[f for f in file_path]
#print(file_names[0])
#print(file_names[0].split('_'))
#print(file_names[0].split('_')[:3])
#print('_'.join(file_names[0].split('_')[:3]))

file_names=pd.Series(file_names)
file_names_updated=file_names.apply(lambda x:'_'.join(x.split('_')[:3]))
file_names_updated=list(file_names_updated)
#print(file_names_updated)

#print(patient_details_df.columns.tolist())

matching_rows = patient_details_df[patient_details_df['Subject'].isin(file_names_updated)]
diagnostic_count=matching_rows['Group'].value_counts()
print(diagnostic_count)