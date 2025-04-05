import os
import pydicom

def sort_and_rename_dicoms(root_dir):
    """
    Sort and rename the DICOM files for per visit using the naming convention

    Input:
        root_dir (str): folder containing the DICOM dataset
    """

    #Path to go through in the given directory: go through all folders and subfolders
    for subdir, _, files in os.walk(root_dir):

        #only include files that are DICOM files
        dicom_files = [f for f in files if f.lower().endswith('.dcm')]

        dicoms = []                                         #initialized list to hold (instance #, file pat)
        for f in dicom_files:                               #go through the DICON metadata
            path = os.path.join(subdir, f)                  #full file path
            try:
                read_dicom = pydicom.dcmread(path)          #load the dicome file
                dicoms.append((read_dicom.InstanceNumber, path))    #add instance # and path to the list
            except Exception as e: 
                print(f" Can't read file {path}: {e}")      #error when reading file fails
        
        if not dicoms:
            continue                                        # if current folder has no DICOM files, skip the folder

        dicoms.sort(key = lambda x: x[0])                   #sort the dicoms by the slice order

        subject_ID = next(p for p in subdir.split(os.sep) if "_S_" in p)    #pull subject_id, based off of '_S_' in the ID

        date = next(p[:10] for p in subdir.split(os.sep) if p.count('-') == 2)      #pull the date of the scan based off of the 2 '-' in the date

        for i, (_, path) in enumerate(dicoms, start = 1):           #for all of the dicom files in current path
            new_file_name = f"{subject_ID}_{date}_Slice_{i}.dcm"    #rename using convention: SubjectID_Date_Slice_i.dcm
            new_path = os.path.join(subdir, new_file_name)
            if path != new_path:                                    #skip renaming the file if it has already been renamed and ordered
                os.rename(path, new_path)

        print(f" Renamed {len(dicoms)} slices in {subdir}")

sort_and_rename_dicoms("Aggregated Dataset")