import os 
import random
import shutil
import re

def patient_split_train_test(input_dir, output_dir, train_ratio = 0.8, seed = 13):
    """
    Splits dataset of DICOM files into training and test sets based off of patient ID

    Input:
        input_dir (str): path to the DICOM dataset
        output_dir (str): output path for the train/test folders
        train_ratio (float): ratio of data in train/test set
        seed (int): random seed
    """

    random.seed(seed)           #set the random seed
    
    subject_ID_to_files = {}    #initialize dictionary to map subject IDs to files

    for filename in os.listdir(input_dir):      #loop through all files in the input directory
        if not filename.endswith(".dcm"):       #skip if the file isn't DICOM format
            continue 

        pattern = re.search(r"\d{3}_S_\d{4}", filename)         #extract patient ID
        if not pattern:                                         #skip if patient ID doesn't follow specified format
            continue

        subject_ID = pattern.group()                                            #extract subject_ID if matches specified format
        subject_ID_to_files.setdefault(subject_ID, []).append(filename)         #add filename to list for specific subject ID
        
    subjects = list(subject_ID_to_files.keys())     #convert subject_ID dictionary to a list
    random.shuffle(subjects)                        #shuffle subject_ID, want randomness for train/test sets

    split_index = int(len(subjects) * train_ratio)  #calculate list index to split groups into train/test
    train_subjects = set(subjects[:split_index])    #first half = train set
    test_subjects = set(subjects[split_index:])     #second half = test set

    train_dir = os.path.join(output_dir, "train")   #create the training output directory if doesn't already exist
    os.makedirs(train_dir, exist_ok = True)
    test_dir = os.path.join(output_dir, "test")     #create the testing output directory if doesn't already exist
    os.makedirs(test_dir, exist_ok = True)

    for subj in train_subjects:                     #copy files for specific subjects to the training set
        for filename in subject_ID_to_files[subj]:
            src = os.path.join(input_dir, filename)
            dst = os.path.join(train_dir, filename)
            shutil.copy2(src, dst)

    for subj in test_subjects:                      #copy files for specific subjects to the test set
        for filename in subject_ID_to_files[subj]:
            src = os.path.join(input_dir, filename)
            dst = os.path.join(test_dir, filename)
            shutil.copy2(src, dst)


    print(f"Split: {len(train_subjects)} patients in train, {len(test_subjects)} in test.")    
    print(f"{len(os.listdir(train_dir))} files in train, {len(os.listdir(test_dir))} in test")

patient_split_train_test("Midline Dataset", "Midline Train Test")