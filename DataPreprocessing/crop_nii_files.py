import numpy as np
import nibabel as nib
import os

def process_nii(data_path, mask_path, output_path):
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Get sorted file lists
    data_files = sorted([f for f in os.listdir(data_path) if f.endswith(".nii")])
    mask_files = sorted([f for f in os.listdir(mask_path) if f.endswith(".nii")])
    count = 0
    
    for data_file in data_files:
        if count < 14965:
            count+=1
            continue
        else:
            corresponding_mask = next((m for m in mask_files if data_file.split(".zip")[0].find(m.split(".nii")[0])> -1 ), None)
            #  1000099_20208_2_0.zip.nii 1000099_20208_2.nii
            if corresponding_mask:
                count+=1
                print(f"This is the {count}th matched file")
                data_nii = nib.load(os.path.join(data_path, data_file))
                mask_nii = nib.load(os.path.join(mask_path, corresponding_mask))
            
                data = data_nii.get_fdata()
                mask = mask_nii.get_fdata()
            
                # Skip images with depth greater than 50
                if data.shape[2] > 50:
                    print(f"Skipping {data_file} due to depth > 50")
                    continue
            
                # Apply mask
                new_data = data * mask
            
                # Extract nonzero region
                nonzero_coords = np.argwhere(new_data > 0)
                min_coords = np.min(nonzero_coords, axis=0)
                max_coords = np.max(nonzero_coords, axis=0)
            
                cropped_data = new_data[min_coords[0]:max_coords[0]+1, 
                                    min_coords[1]:max_coords[1]+1, 
                                    :]
            
                # Ensure cropped data fits within (80,80,50)
                crop_shape = cropped_data.shape
                crop_x = min(crop_shape[0], 80)
                crop_y = min(crop_shape[1], 80)
                cropped_data = cropped_data[:crop_x, :crop_y, :]
            
                # Pad to (90, 90, 50)
                padded_data = np.zeros((80, 80, 50))
                start_x = (80 - crop_x) // 2
                start_y = (80 - crop_y) // 2
            
                padded_data[start_x:start_x+crop_x, start_y:start_y+crop_y, :] = cropped_data
            
                # Save result
                new_nii = nib.Nifti1Image(padded_data, affine=data_nii.affine)
                output_file = os.path.join(output_path, f"cropped_{data_file}")
                nib.save(new_nii, output_file)
                print(f"Processed and saved: {output_file}")

# Example usage
data_path = "" # the path you save the original nii files
mask_path = "" # the path you save the mask nii files
output_path = "" # the output path which you will save the nii files with heart area with the resolution of 80*80*50
process_nii(data_path, mask_path, output_path)
