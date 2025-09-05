import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import os
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim_loss
from tqdm import tqdm
import random
from model import CNN3D

save_path="" #model save path
class NiiDataset(Dataset):
    def __init__(self, file_paths):
        """
        Initialize the dataset with a list of file paths.
        Args:
        - file_paths: List of paths to .nii files.
        """
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Load the .nii file and return the 3D data as a tensor.
        Args:
        - idx: Index of the file in the dataset.
        Returns:
        - Tensor of shape (1, 128, 128, 50) representing the 3D data.
        """
        nii_path = self.file_paths[idx]
        img = nib.load(nii_path).get_fdata()
        img = np.expand_dims(img, axis=0)  # Add channel dimension
        img = img.astype(np.float32) / np.max(img)  # Normalize to [0, 1]
        return torch.tensor(img)

# Function to prepare dataset
def prepare_dataset(folder_path, test_size=0.1, batch_size=16):
    """
    Prepare training and testing datasets from a folder of .nii files.
    Args:
    - folder_path: Path to the folder containing .nii files.
    - test_size: Fraction of data to use for testing (default 0.2).
    - batch_size: Batch size for DataLoader (default 16).
    Returns:
    - train_loader: DataLoader for training.
    - test_loader: DataLoader for testing.
    """
    # Gather all .nii file paths
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.nii')]

    # Split into training and testing sets
    train_files, test_files = train_test_split(file_paths, test_size=test_size, random_state=42)

    # Create datasets
    train_dataset = NiiDataset(train_files)
    test_dataset = NiiDataset(test_files)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
        
def save_slice_jig(real_nii, fake_nii, save_path, epoch):
    """
    Save a jig of the 25th slice of real and fake NIfTI images.

    Args:
    - real_nii: Real 3D NIfTI tensor (depth x height x width).
    - fake_nii: Fake 3D NIfTI tensor (depth x height x width).
    - save_path: Path to save the output jig.
    - epoch: Current epoch number.
    """
    os.makedirs(save_path, exist_ok=True)

    # Select the 25th slice (index 24) and detach tensors
    real_slice = real_nii[:, :, 24].cpu().detach().numpy()
    fake_slice = fake_nii[:, :, 24].cpu().detach().numpy()

    # Create a plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(real_slice, cmap="gray")
    axes[0].set_title("Real Slice (25th)")
    axes[0].axis("off")

    axes[1].imshow(fake_slice, cmap="gray")
    axes[1].set_title("Fake Slice (25th)")
    axes[1].axis("off")

    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"epoch_{epoch}_slice_25.png"))
    plt.close()
    
def apply_multi_mask(input_data, mask_size=(50, 8, 8), num_masks=75, device="cuda"):
    """
    Apply multiple random 3D masks of fixed size to the input data.
    Each mask zeroes out a random block of shape (depth, height, width).

    :param input_data: Tensor of shape (batch_size, channels, depth, height, width)
    :param mask_size: Tuple (mask_d, mask_h, mask_w) defining size of each mask
    :param num_masks: Number of random masks per sample
    :param device: Torch device
    :return: masked_input (Tensor), mask (Tensor with 1s where data is kept and 0s where masked)
    """
    if input_data.shape[4] < input_data.shape[2]:  # assume D is the smallest dim
        input_data = input_data.permute(0, 1, 4, 2, 3)  # (B, C, H, W, D) -> (B, C, D, H, W)
        permuted = True
    batch_size, channels, depth, height, width = input_data.size()
    mask_d, mask_h, mask_w = mask_size

    # Initialize mask with ones (keep data)
    mask = torch.ones((batch_size, 1, depth, height, width), device=device)

    for b in range(batch_size):
        used_coords = set()
        for _ in range(num_masks):
            # Ensure the mask fits within the data dimensions
            max_d = depth - mask_d
            max_h = height - mask_h
            max_w = width - mask_w

            # Avoid overlap (optional - not strictly enforced here)
            while True:
                d = random.randint(0, max_d)
                h = random.randint(0, max_h)
                w = random.randint(0, max_w)
                coord = (d, h, w)
                if coord not in used_coords:
                    used_coords.add(coord)
                    break

            # Apply the mask (set values to 0 in the mask tensor)
            mask[b, :, d:d+mask_d, h:h+mask_h, w:w+mask_w] = 0

    # Apply mask to the input
    masked_input = input_data * mask
    # Restore original shape if it was permuted
    if permuted:
        masked_input = masked_input.permute(0, 1, 3, 4, 2)  # back to (B, C, H, W, D)

    return masked_input, mask
       
def train_model(model, train_loader, test_loader, device, save_path, num_epochs=100, log_file="./train_log.txt"):
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model.to(device)
    os.makedirs(save_path, exist_ok=True)
    if os.path.exists(log_file):
        os.remove(log_file)
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0
        train_loader_tqdm = tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=True)

        for batch in train_loader_tqdm:
            try:
                #batch = batch.to(device, non_blocking=True)
                batch = batch.to(device, non_blocking=True)
                masked_input, mask = apply_multi_mask(batch, mask_size=(50,8,8),num_masks=75, device=device)
                optimizer.zero_grad()
                reconstruction, _ = model(masked_input)
                loss_mse = criterion(reconstruction, batch)
                ssim = ssim_loss(reconstruction, batch, data_range=1.0)
                loss = loss_mse - 0.01 * ssim
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_loader_tqdm.set_postfix(loss=loss.item())
            except RuntimeError as e:
                print(f"Error processing batch: {e}")
                continue  # Skip the problematic batch

        model.eval()
        test_loss = 0
        test_loader_tqdm = tqdm(test_loader, desc=f"Testing Epoch {epoch}", leave=True)
        with torch.no_grad():
            for batch in test_loader_tqdm:
                try:
                    batch = batch.to(device, non_blocking=True)
                    masked_input, mask = apply_multi_mask(batch, mask_size=(50,8,8),num_masks=75, device=device)
                    reconstruction, _ = model(masked_input)
                    loss_mse = criterion(reconstruction, batch)
                    ssim = ssim_loss(reconstruction, batch, data_range=1.0)
                    loss = loss_mse - 0.01 * ssim
                    test_loss += loss.item()
                    test_loader_tqdm.set_postfix(loss=loss.item())
                except RuntimeError as e:
                    print(f"Error processing test batch: {e}")
                    continue  # Skip the problematic batch

        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        log_message = (f"Epoch [{epoch}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, "
                       f"Test Loss: {avg_test_loss:.4f}\n")
        print(log_message.strip())
        with open(log_file, "a") as f:
            f.write(log_message)

        if epoch % 1 == 0:
            save_slice_jig(batch[0, 0], reconstruction[0, 0], save_path, epoch)
        if epoch % 1 == 0:
            model_save_path = os.path.join(save_path, f"model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at {model_save_path}")

if __name__ == "__main__":
    # Model, DataLoader, and device setup
    folder_path = "" # training data folder
    train_loader, test_loader = prepare_dataset(folder_path)
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    num_epochs = 50
    model = CNN3D(in_channels=1, latent_dim=256)   
    train_model(model, train_loader, test_loader, device, save_path, num_epochs=num_epochs, log_file="./train_log.txt")
