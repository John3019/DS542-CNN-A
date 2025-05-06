import os
import re

def count_patient_visits(root_dir):
    visit_count = 0
    visits = set()                              #to track (subject_ID, date) pairs

    for filename in os.listdir(root_dir):       #loop through all files in folder
        if not filename.endswith(".dcm"):       #skip files that are not DICOM format
            continue

        subject_match = re.search(r"\d{3}_S_\d{4}", filename)       #extract subject_ID from filename
        date_match = re.search(r"\d{4}-\d{2}-\d{2}", filename)      #extract date from filename

        if not subject_match or not date_match:     #skip file if doesn't follow naming convention
            continue

        subject_ID = subject_match.group()          #matched subject_ID
        date = date_match.group()                   #matched date

        visit_key = (subject_ID, date)              #tuple (subject_ID, date)
        if visit_key not in visits:                 #if this unique visit isn't already in the set
            visits.add(visit_key)                   #add it to the set
            visit_count += 1                        #increment the counter

    print(f"Total number of patient visits: {visit_count}")
    return visit_count


count_patient_visits("Midline Dataset")
