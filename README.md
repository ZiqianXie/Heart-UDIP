The codes in this repo is for our paper "Uncovering genetic architecture of the heart via genetic association studies of unsupervised deep learning derived endophenotypes"
<img width="1125" height="633" alt="image" src="https://github.com/user-attachments/assets/26fa0d37-f63f-45e5-9f40-20bbdef01f37" />

Main structure of this repo

1.data preprocessing pipe line which includes the functions like dicom2nii, nii_cropping, registration_template and image_registrations with ANTs. First, you need to run dicom2nii to conver all the dicom files to nii format. 2.In this part, you need to annoted around 100 nii files to fine-tune the nnUnet to get the segmentation results of all the files you have. after that, you can ust the nii_cropping to get the heart area only nii files with the resolution of 80 * 80 * 50. 3. The heart registration template is built on 150 ramdonly selected nii files with the resolution of 80 * 80 * 50. 4. Use the template to register all the nii files (80 * 80 * 50).

2.the training codes of our cine MAE. We use 10000 cases to train the cineMAE, leaving the rest ones as test data.

3.the evaluation codes like SSIM and PNSR calculation.

4.the interpretabilty codes like our Perturbation-based image reconstruction.

5.GWAS and PostGWAS code.The complete GWAS summary statistics have been deposited in the GWAS Catalog (accession GCP001376).

Citations: Pending
