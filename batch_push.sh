#!/bin/bash

# CONFIG: Set your batch size
BATCH_SIZE=5000

# CONFIG: Set your folder (train or test)
DATA_FOLDER="Data/train"

# Counter
counter=0

# Start looping over all DICOM files
for file in $(find "$DATA_FOLDER" -type f -name "*.dcm"); do
    git add "$file"
    counter=$((counter + 1))

    # When batch size is reached, commit and push
    if [ "$counter" -eq "$BATCH_SIZE" ]; then
        git commit -m "Adding batch of $BATCH_SIZE DICOMs from $DATA_FOLDER"
        git push
        counter=0
    fi
done

# Push any leftover files
if [ "$counter" -gt 0 ]; then
    git commit -m "Adding final batch of DICOMs from $DATA_FOLDER"
    git push
fi

echo "✅ Finished pushing all batches from $DATA_FOLDER!"
