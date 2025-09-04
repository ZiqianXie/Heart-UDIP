import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import os
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim_loss
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import time
from model import CNN3D 

class NiiDataset(Dataset):
    def __init__(self, folder_path):
        self.file_paths = []
        
        # Filter out bad files
        for f in os.listdir(folder_path):
            if f.endswith('.nii'):
                file_path = os.path.join(folder_path, f)
                img = nib.load(file_path).get_fdata(dtype=np.float32)
                
                if np.max(img) > 0:  # Exclude zero max images
                    self.file_paths.append(file_path)
                else:
                    print(f"Skipping {f}: max value is zero")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        nii_path = self.file_paths[idx]
        img = nib.load(nii_path).get_fdata(dtype=np.float32)

        img = np.expand_dims(img, axis=0)
        img = img / np.max(img)  # Safe now, since max is non-zero
        return torch.from_numpy(img), os.path.basename(nii_path)  
      
# Function to extract latent features
def extract_latent_features(model, dataloader, device, save_path):
    model.eval()
    results = []
    with torch.no_grad():
        for images, file_names in dataloader:
            images = images.to(device)
            _, latents = model(images)  # Get the latent features
            #print(f"latents shape is : {latents.shape}")
            latents = latents.cpu().numpy()
            
            for i, file_name in enumerate(file_names):
                # Extract patient ID and visit
                patient_id = file_name.split("_")[2]
                visit = file_name.split("_")[4]
                # Combine patient ID, visit, and latent features
                results.append([patient_id, visit] + latents[i].tolist())
    
    # Save to CSV
    df = pd.DataFrame(results, columns=["Patient_ID", "Visit"] + [f"Feature_{i+1}" for i in range(latents.shape[1])])
    df.to_csv(save_path, index=False)
    print(f"Latent features saved to {save_path}")

# Main script
def main():
    start = time.time()
    folder_path = ""  # test nii files folder
    save_path = ""  # Output CSV file like .../.../UKB_60000_features.csv
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    # Load the pre-trained model
    model = CNN3D(in_channels=1, latent_dim=256)
    model.load_state_dict(torch.load("/data/..../model.pth", map_location=device))
    model.to(device)
    #model.summary()
    end = time.time()
    print(f"it takes {end - start} seconds to run this section")
    # Prepare dataset and dataloader
    dataset = NiiDataset(folder_path)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    # Extract and save latent features
    extract_latent_features(model, dataloader, device, save_path)

if __name__ == "__main__":
    main()
