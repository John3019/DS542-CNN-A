import os
import shutil
import re

def pull_midline_scan(root_dir, output_dir):
    """
    Find the middle DICOM slice per patient 

    Input:
        root_dir (str): path to root folder with all DICOM dataset
        output_dir (str): path to output folder where midline slices will go
    """
    
    completed_subjects = set()                          #track patients (subject_ID, date) that have already been added to the lis
    slice_number = re.compile(r"Slice_(\d+)\.dcm$")     #extract the slice number from the file name

    for subdir, _, files in os.walk(root_dir):                      #go through all subdirectories in the root_dir
        dicom_files = [f for f in files if f.endswith('.dcm')]      #gather a list of all.dcm files in subdir
        if not dicom_files:                                         #skip if the files is not DICOM
            continue

        try:
            subject_ID = next(p for p in subdir.split(os.sep) if "_S_" in p)    #extract subject_ID from the file name
        except StopIteration:                                                   #skip if the subject_ID isn't in file name
            continue 

        try: 
            date = next(p[:10] for p in subdir.split(os.sep) if p.count('-') == 2)      #extract date from file name
        except StopIteration:                                                           #skip if the date isn't in the file name
            continue

        subject_keys = (subject_ID, date)           #combine subject_ID and visit date into key
        if subject_keys in completed_subjects:      #if key already exists, skip it
            continue

        dicom_sorted = sorted(dicom_files, key = lambda x: int(slice_number.search(x).group(1)))        #sort the DICOM files into order

        middle_index = len(dicom_sorted) // 2               #get the middle index of the list
        middle_file = dicom_sorted[middle_index]            #get the middle slice from the sorted list
        middle_path = os.path.join(subdir, middle_file)     #build source path
        output_path = os.path.join(output_dir, middle_file) #build output path
        shutil.copy2(middle_path, output_path)              #copy slive to output directory

        print(f"{subject_ID} ({date}): copied {middle_file} to {output_path}")
        completed_subjects.add(subject_keys)

pull_midline_scan("Aggregated Dataset", "Midline Dataset")