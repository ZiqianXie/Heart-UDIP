import ants
import os
import numpy as np

nii_folder = '' # the path you save the nii files with resolution of 80*80*50
fixed_image = ants.image_read('./template_150_zfixed.nii')
out_folder = ''
warp_folder = ''  # Folder to save warp fields
affine_folder = ""

# Create output directories if they don't exist
os.makedirs(out_folder, exist_ok=True)
os.makedirs(warp_folder, exist_ok=True)
print(f"Found {len(nii_files)} .nii files for registration.")


# List all .nii files in the folder
nii_files = [f for f in os.listdir(nii_folder) if f.endswith('.nii')]
print(f"Found {len(nii_files)} .nii files for registration.")
count = 0

for each_file in nii_files:
    count += 1
    if count < 0:
        continue
    else:
        print(f"\nProcessing {count}-th file: {each_file}")

        img_path = os.path.join(nii_folder, each_file)
        raw_imgs = ants.image_read(img_path)

        # Check if image is not all zeros
        if np.all(raw_imgs.numpy() == 0):
            print("Skipped: Image has all zero values.")
            continue

        try:
            registration = ants.registration(
                fixed=fixed_image,
                moving=raw_imgs,
                type_of_transform='Affine',
                outprefix ='./ant_tmp_files/',
                restrict_transformation=(1,1,0,1,1,0,0,0,0,1,1,0)
            )

            registered_image = registration['warpedmovout']
            out_filename = 'registered_' + each_file
            out_path = os.path.join(out_folder, out_filename)
            registered_image.to_file(out_path)
            print("Successfully registered and saved.")

        except Exception as e:
            print(f"Registration failed for {each_file}: {e}")

print("\nRegistration complete!")
