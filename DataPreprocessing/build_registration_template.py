import ants
import os
import os

def get_partial_file_list(directory, max_files=152):
    partial_list = []
    image_list  = []
    with os.scandir(directory) as entries:
        for i, entry in enumerate(entries):
            if i >= max_files:
                break
            if entry.is_file():
                partial_list.append(entry.name)
                image_list.append(ants.image_read(os.path.join(directory,entry)))
    return image_list

# Example usage:
directory_path = '' # the path you save the nii files with the resolution of 80*80*50
partial_images = get_partial_file_list(directory_path, max_files=150)

template_image = ants.build_template(image_list=partial_images,useNoRigid=False,restrict_transformation=(1,1,0,1,1,0,0,0,0,1,1,0))
out_folder = '' # the path you will save the template
out_filename = 'template_150_zfixed.nii'
out_path = os.path.join(out_folder, out_filename)
print(out_path)
template_image.to_file(out_path)
print("finished")
