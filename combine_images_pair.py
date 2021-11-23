from pathlib import Path
import cv2
from PIL import Image
import numpy as np
import os
import glob
import random
import shutil
# HE_path = Path("/data04/shared/mahmudul/HE_to_IHC/data/HE_to_IHC_patches/925-multires_paired_HE")
# IHC_path = Path("/data04/shared/mahmudul/HE_to_IHC/data/HE_to_IHC_patches/925-multires_paired_mIHC")
# output_dir = Path("/data04/shared/mahmudul/HE_to_IHC/data/HE_to_IHC_patches/paired_images")
# HE_files = HE_path.glob("*.png")
# IHC_files = IHC_path.glob("*.png")
# HE_dict = {}
# IHC_dict = {}
# for f in HE_files:
#     HE_dict[f.name.split('_')[0]] = f

# for f in IHC_files:
#     ihc_image = cv2.imread(str(f))
#     filename = f.name.split('_')[0]
#     he_image = cv2.imread(str(HE_dict[filename]))
#     he_image_rgba = cv2.cvtColor(he_image, cv2.COLOR_BGR2BGRA)
#     ihc_image_rgba = cv2.cvtColor(ihc_image, cv2.COLOR_BGR2BGRA)
#     if np.mean(he_image_rgba[3]) > 223 or np.mean(ihc_image_rgba[3]) > 223:
#         continue
#     merged_img = cv2.hconcat([he_image,ihc_image])
#     cv2.imwrite(str(output_dir / (filename + '.png')),merged_img)

data_dir = Path("/data04/shared/mahmudul/HE_to_IHC/data/HE_to_IHC_patches/paired_images")
training_dir = Path("/data04/shared/mahmudul/HE_to_IHC/data/HE_to_IHC_patches/training_data")
train_dir = training_dir / 'train'
val_dir = training_dir / 'val'
test_dir = training_dir / 'test'
val_split = 0.1
test_split = 0.1

if not train_dir.exists():
    os.mkdir(str(train_dir))

if not val_dir.exists():
    os.mkdir(str(val_dir))

if not test_dir.exists():
    os.mkdir(str(test_dir))

data_list = glob.glob(str(data_dir / '*.png'))
val_split = int(len(data_list)*val_split)
test_split = int(len(data_list)*test_split)
train_split = len(data_list) - (val_split + test_split)
random.shuffle(data_list)
train_list = data_list[:train_split]
val_list = data_list[train_split:train_split+val_split]
test_list = data_list[train_split+val_split:]

for f in train_list:
    shutil.copy2(f,train_dir)

for f in val_list:
    shutil.copy2(f,val_dir)

for f in test_list:
    shutil.copy2(f,test_dir)