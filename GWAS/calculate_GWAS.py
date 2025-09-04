############################################# phenotype extractions
import pandas as pd
import os
heart_endo = pd.read_csv('.../UKB_60000_features.csv')
len1=len(heart_endo)
heart_endo= heart_endo[~heart_endo['Patient_ID'].duplicated(keep=False)] # removing duplicate patient_id rows (even first occurrences)
len2=len(heart_endo)
print(len1-len2)

heart_endo.rename(columns={'Patient_ID': 'FID'}, inplace=True)
heart_endo['IID'] = heart_endo['FID']

cols = ['FID', 'IID'] + heart_endo.columns.tolist()[2:-1]
heart_endo = heart_endo[cols]

output_dir = '/UKB_60000_phenos/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for i in range(1,257):
    df = heart_endo[['FID', 'IID', 'Feature_'+str(i)]]
    file_name = f'Feature_{i}'
    df.to_csv(output_dir + file_name, sep=' ', index=False)
    
#############################################
############################################# gwas
import pandas as pd
import os
pheno=pd.read_csv('.../UKB_60000_features.csv')
pheno_columns= list(pheno.columns[2:])
output_dir = '/UKB_60000_fastGWA/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


for i in pheno_columns:
    cmd = f"..../gcta/gcta-1.94.1-linux-kernel-3-x86_64/gcta-1.94.1 --maf 0.01 --bfile ../Heart/heart_pheno --grm-sparse ../gcta/ukb_grm --fastGWA-mlm --pheno .../Heart/UKB_60000_phenos/{i} --covar ../Heart/ccovar --qcovar ../Heart/qcovar --thread-num 256 --seed 0 --out .../Heart/UKB_60000_fastGWA/{i}"
    os.system(cmd)






