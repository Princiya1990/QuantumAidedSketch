import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.decomposition import PCA

# ---------------------------
# 1. Setup CNN (ResNet18 feature extractor)
# ---------------------------
resnet = models.resnet18(pretrained=True)
resnet.fc = torch.nn.Identity()   
resnet.eval()

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------------------------
# 2. Read images from folder
# ---------------------------
INPUT_FOLDER = "superresolved_sketches"   
OUTPUT_CSV   = "features.csv"

image_paths = glob.glob(os.path.join(INPUT_FOLDER, "*.png")) + \
              glob.glob(os.path.join(INPUT_FOLDER, "*.jpg")) + \
              glob.glob(os.path.join(INPUT_FOLDER, "*.jpeg"))

features_list = []
file_names = []

# ---------------------------
# 3. Extract CNN features
# ---------------------------
with torch.no_grad():
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0)
        feat = resnet(x).numpy().flatten()   # 512-dim vector
        features_list.append(feat)
        file_names.append(os.path.basename(img_path))

features_array = np.array(features_list)  # shape: (N, 512)

# ---------------------------
# 4. Dimensionality Reduction (PCA → 6 dims)
# ---------------------------
pca = PCA(n_components=6)
features_reduced = pca.fit_transform(features_array)  # shape: (N,6)

# ---------------------------
# 5. Save to CSV
# ---------------------------
df = pd.DataFrame(features_reduced, columns=[f"feat_{i+1}" for i in range(6)])
df.insert(0, "filename", file_names)  # keep track of which image
df.to_csv(OUTPUT_CSV, index=False)

print(f"[INFO] Extracted features saved to {OUTPUT_CSV}")
