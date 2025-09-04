import os
import random
import torch
import nibabel as nb
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from scipy.stats import ttest_rel
from torch.utils.data import DataLoader, Dataset
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
from scipy.ndimage import gaussian_filter
from model import CNN3D 


folder_path = "" # the test data folder
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

class NiftiDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        nifti_img = nb.load(self.file_paths[idx])
        img_data = nifti_img.get_fdata()
        img_tensor = torch.tensor(img_data, dtype=torch.float32).unsqueeze(0)  # Add channel dim
        return img_tensor, self.file_paths[idx]

# Load random 500 nii files from a folder
def load_nii_files(folder, num_files=500):
    all_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".nii") or f.endswith(".nii.gz")]
    selected_files = random.sample(all_files, num_files)
    return selected_files
        
# Extract enc2 and latent representation
def extract_features(dataloader, device):
    model.eval()
    enc2_list, latent_list, filenames = [], [], []
    with torch.no_grad():
        for img, filename in tqdm(dataloader):
            img = img.to(device)
            enc1 = model.encoder_relu1(model.encoder_norm1(model.encoder_conv1(img)))
            enc2 = model.encoder_relu2(model.encoder_norm2(model.encoder_conv2(enc1)))
            enc3 = model.encoder_relu3(model.encoder_norm3(model.encoder_conv3(enc2)))
            
            latent = model.encoder_fc(model.flatten(enc3))
            enc2_list.append(enc2.cpu().numpy()) 
            latent_list.append(latent.cpu().numpy())
            filenames.extend(filename)
    return np.vstack(enc2_list), np.vstack(latent_list), filenames


# Save latent features to CSV
def save_latent_to_csv(latent, filenames, output_csv):
    latent = latent.squeeze(1)
    df = pd.DataFrame(latent)
    #latent_list.append(latent.cpu().numpy().squeeze(1))
    df.insert(0, "Filename", filenames)
    
    df.to_csv(output_csv, index=False)


# Decode function using enc2 and latent
def decode(some_enc2, latent):
    with torch.no_grad():
        print(f"the shape of latent is {latent.shape}")
        dec_input = model.decoder_fc(torch.tensor(latent, dtype=torch.float32))       
        dec_input = dec_input.view(-1, 64, 10, 10, 7)
        print(f"the shape of dec_input is {dec_input.shape}")
        #dec_input = model.unflatten(dec_input)
        #enc2 = torch.tensor(enc2, dtype=torch.float32).to(device)       
        dec1 = model.decoder_relu1(model.decoder_norm1(model.decoder_conv1(dec_input)))
        print(f"the shape of dec1 is {dec1.shape}")
        dec1 = 0.5 * torch.tensor(some_enc2, dtype=torch.float32) + 0.5 * dec1
        dec2 = model.decoder_relu2(model.decoder_norm2(model.decoder_conv2(dec1)))
        dec3 = model.decoder_sigmoid(model.decoder_conv3(dec2))
    return dec3

# Save results as NIfTI
def save_for_viz(attribution, filename, affine_save):
    nifti_img = nb.Nifti1Image(attribution, affine_save)
    nb.save(nifti_img, f"/data/.../inter_pre/{filename}.nii.gz")
    
def process_data02(enc2, latent, device,dim, affine_ori,output_prefix="paired_ttest_T1"):
    original_sd, perturb_sd = [], []   

    
    for i in tqdm(range(0,500,10)):
        first_enc2 = enc2[i:i+10, :]
        first_latent = latent[i:i+10, :].copy()
        # Convert to torch tensor
        first_enc2_tensor = torch.from_numpy(first_enc2).float().to(device)
        first_latent_tensor = torch.from_numpy(first_latent).float().to(device)        
        recon1 = decode(first_enc2_tensor, first_latent_tensor)
        sd = np.std(pd.DataFrame(latent[:500,:]).iloc[:,dim])
        first_latent[:, dim] += sd
        first_latent_tensor_sd = torch.from_numpy(first_latent).float().to(device)
        recon2 = decode(first_enc2_tensor, first_latent_tensor_sd)
        original_sd.extend(recon1.detach().cpu().numpy())
        perturb_sd.extend(recon2.detach().cpu().numpy())
        
    original_sd = [np.squeeze(i) for i in original_sd]
    perturb_sd = [np.squeeze(i) for i in perturb_sd]
              
    # Use only the first patient's enc2 and latent for visualization
    t_sd = ttest_rel(original_sd, perturb_sd, axis=0, nan_policy="omit")
    t_sd = abs(t_sd[0])
    t_sd = gaussian_filter(t_sd, sigma=3)
    print(f"t_sd shape is: {t_sd.shape}")
    save_prefix = str(dim)+"_"+output_prefix
    save_for_viz(t_sd, save_prefix,affine_ori)


model = CNN3D(in_channels=1, latent_dim=256)
model.load_state_dict(torch.load("/data/.../model.pth", map_location=device))
model.to(device)
# Load data and extract features
nii_files = load_nii_files(folder_path)
affine_matrix = nb.load(nii_files[0]).affine
dataset = NiftiDataset(nii_files)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

enc2, latent, filenames = extract_features(dataloader, device)

print(f"the shape of enc2 is: {enc2.shape}")
print(f"the shape of latent is: {latent.shape}")

for i in tqdm(range(0,256,1)):
    process_data02(enc2,latent,device,i,affine_matrix)
