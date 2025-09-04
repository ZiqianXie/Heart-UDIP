import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os 
import nibabel as nib

def plot_nii_slices_with_colorbar(nii_path):
    # Load the image
    img = nib.load(nii_path)
    data = img.get_fdata()

    # Check if 3D
    if data.ndim != 3:
        raise ValueError("Expected a 3D NIfTI image")

    num_slices = data.shape[2]

    # Plot each slice
    for i in range(0,num_slices,5):
        slice_data = data[:, :, i]

        plt.figure(figsize=(6, 5))
        im = plt.imshow(slice_data.T, cmap='plasma', origin='lower')  # transpose to match anatomical orientation
        plt.title(f"Slice {i + 1}")
        plt.colorbar(im, label='Intensity',fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

def plot_nii_slices_with_colorbar_save(nii_path, output_folder):
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load the image
    img = nib.load(nii_path)
    data = img.get_fdata()

    # Check if 3D
    if data.ndim != 3:
        raise ValueError("Expected a 3D NIfTI image")

    num_slices = data.shape[2]

    # Plot and save each slice
    for i in range(0, num_slices, 1):
        slice_data = data[:, :, i]
        slice_data[slice_data < 5] = 0

        plt.figure(figsize=(6, 5))
        #im = plt.imshow(slice_data.T, cmap='plasma', origin='lower')  # transpose to match anatomical orientation
        im = plt.imshow(slice_data, cmap='plasma', origin='upper')
        plt.title(f"Slice {i}",fontsize=16)
        # Create the colorbar and set its label with a specific font size
        cbar = plt.colorbar(im)
        cbar.set_label('Intensity', size=20) # You can adjust the size '14' as needed
        
        plt.axis('off')
        plt.tight_layout()
        #plt.colorbar(im, label='Intensity')
        #plt.axis('off')
        #plt.tight_layout()

        # Save to file
        output_path = os.path.join(output_folder, f"slice_{i:03d}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Saved slices to: {output_folder}")
# Example usage:
for case_number in range(256):
    case_name = "/data/.../"+str(case_number)+"_paired_ttest_T1.nii.gz"
    folder_name = '/data/../feature_'+str(case_number)
    plot_nii_slices_with_colorbar_save(case_name,folder_name)
