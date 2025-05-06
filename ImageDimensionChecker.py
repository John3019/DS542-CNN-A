import numpy as np
import pydicom

dcm=pydicom.dcmread("/projectnb/dl4ds/students/atuladas/DS542-CNN-A/Midline Train Test/train/002_S_0295_2012-05-10_Slice_86.dcm")

pixel_array=dcm.pixel_array
print("Image shape:",pixel_array.shape)