import os
import pydicom
import numpy as np
import nibabel as nib
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2


def load_dicom_images(folder_path, error_log_path):
    dicom_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.dcm')]
    
    # Read all DICOM files and sort by InstanceNumber
    dicom_data = []
    for file in dicom_files:
        try:
            dicom = pydicom.dcmread(file)
            instance_number = getattr(dicom, "InstanceNumber", float('inf'))
            dicom_data.append((file, dicom, instance_number))
        except Exception as e:
            with open(error_log_path, 'a') as error_log:
                error_log.write(f"Error reading file {file}: {str(e)}\n")
            print(f"Error reading file {file}. Logged to {error_log_path}. Continuing...")
    
    # Sort files based on InstanceNumber
    dicom_data.sort(key=lambda x: x[2])
    
    images = []
    target_size = None  # Placeholder for target size 
    
    for i, (file, dicom, _) in enumerate(dicom_data):
        try:
            img = dicom.pixel_array
            
            if i == 0:
                # Set the resolution of the first image as the target size
                target_size = img.shape
                print(f"Using resolution of first image as target size: {target_size}")
            
            # Resize other images to match the target size
            if img.shape != target_size:
                img = cv2.resize(img, (target_size[1], target_size[0]))
            
            images.append(img)
        
        except ValueError as e:
            # Log the file name if a ValueError occurs
            with open(error_log_path, 'a') as error_log:
                error_log.write(f"Error processing file {file}: {str(e)}\n")
            print(f"ValueError encountered for file {file}. Logged to {error_log_path}. Continuing...")
    
    return np.array(images)

def convert_to_nii(images, output_path, error_log_path,current_index):
    image_index = current_index
    try:
        # Transpose the images to the (x, y, z) format required by NIfTI
        images_transposed = np.transpose(images, (1, 2, 0))
        
        # Create a NIfTI image
        nifti_image = nib.Nifti1Image(images_transposed, np.eye(4))
        
        # Save the NIfTI image to the specified path
        nib.save(nifti_image, output_path)
        print(f"NIfTI file has been saved successfully at {output_path}.")
    except ValueError as e:
        # Log the error if a transpose issue occurs
        with open(error_log_path, 'a') as error_log:
            error_log.write(f" {image_index}: Error saving NIfTI file: {str(e)}\n")
        print(f"ValueError encountered during NIfTI conversion. Logged to {error_log_path}. Continuing...")


def main():
    dicom_path= '' # the folder that you save all the subfolders of the dicom files of each patient
    nii_path = '' # the folder that you will save the nii files
    error_log_path = '' # error log file
    all_folders = [folder for folder in os.listdir(dicom_path) if os.path.isdir(os.path.join(dicom_path, folder))]
    count = 0 # the number of cases that you need to process
    for each_folder in all_folders:
        print("This is the {}th folder".format(count))
        
        if count < 0:
            count += 1
            continue
        else:
            count += 1
            nii_name = each_folder+'.nii.gz'
            output_nii_file = os.path.join(nii_path,nii_name)
            full_path = os.path.join(dicom_path,each_folder)
            print("full path is {}".format(full_path))
            #images = load_dicom_images(full_path)
            #convert_to_nii(images, output_nii_file)
            images = load_dicom_images(full_path,error_log_path)
            convert_to_nii(images, output_nii_file,error_log_path,count)
            print(f"NIfTI file has been saved successfully at {output_nii_file}.")
            if count > 10000:
                break
    print(f'There are {count} nii files')   

if __name__ == "__main__":
    main()

