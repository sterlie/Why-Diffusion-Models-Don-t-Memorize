import torch
import pandas as pd
from PIL import Image
import os
import numpy as np

metadata_csv = "Experiments/Data/milk10/MILK10k_Training_Metadata.csv"
image_root = "Experiments/Data/milk10/MILK10k_Training_Input"
output_pth = "Experiments/Data/milk10/MILK10k_Training_Input.pth"

df = pd.read_csv(metadata_csv)
images_dict = {}

for _, row in df.iterrows():
    lesion_id = row['lesion_id']
    isic_id = row['isic_id']
    img_path = os.path.join(image_root, lesion_id, f"{isic_id}.jpg")
    
    if os.path.exists(img_path):
        img = Image.open(img_path).convert('RGB')
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0  # [C,H,W]
        images_dict[isic_id] = img_tensor
    else:
        print(f"Image not found: {img_path}")

os.makedirs(os.path.dirname(output_pth), exist_ok=True)
torch.save(images_dict, output_pth)