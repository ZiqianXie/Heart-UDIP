#############################################
############################################# minP
from pathlib import Path
import os, numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from subprocess import check_output, STDOUT
from itertools import zip_longest
from glob import glob
import pandas as pd



def extract(x, exclusion, out_path):
    # extract rows in the file x not contained in exclusion and save in out_path.
    x = Path(x)
    out_path = Path(out_path)
    cmd = f"awk 'NR == FNR {{ excl[$1]; next }} !(FNR in excl)' {exclusion} {x} > {x.parent/out_path/(x.name.split('.')[0] + '_extracted')}"
    os.system(cmd)
    

def extract_col(x, offset=1):
    out = check_output(f"awk '{{print $(NF-{offset})}}' {x}", universal_newlines=True, shell=True, stderr=STDOUT)
    return np.array(list(map(float, out.strip('\n').split('\n')[1:]))), x

def create_minP(glob_list, pcol=0, mode='min'):
    print(pcol)
    # pcol is indexed from right to left, 0 means last col
    if mode == 'min':
        op = np.argmin
    elif mode == 'max':
        op = np.argmax
    else:
        raise Exception('not implemented')
    batch = 50
    for i in tqdm(range(0, len(glob_list), batch)):
        with Pool(batch) as q:
            result = q.starmap(extract_col, zip_longest(glob_list[i:i+batch], (), fillvalue=pcol))
        pnew, fnew = list(zip(*result))
        if i == 0:
            pnew = np.vstack(pnew)
            idx = op(pnew, 0)
            f = np.array(fnew)[idx]
        else:
            pnew = list(pnew)
            pnew.append(p)
            pnew = np.vstack(pnew)
            idx = op(pnew, 0)
            mask = (idx != (pnew.shape[0]-1))
            f[mask] = np.array(fnew)[idx[mask]]
        p = pnew[idx, np.arange(pnew.shape[1])]
    return p, f
    
p, f = create_minP(glob(".../Heart/UKB_60000_fastGWA/*fastGWA"), 0)
print(f[0])


#template = pd.read_table(f[0]+'A') # change this according to your results.
template = pd.read_table(f[0]) # change this according to your results.
template["P"] = p
template[['CHR', 'SNP', 'POS', 'A1', 'A2', 'AF1', 'N', 'P']].to_csv(".../Heart/minP/GWAS_60000_minP.tsv", sep='\t', index=False)

import csv

def filter_rows_by_p_value(file_path, threshold=1.953125e-10):
    filtered_rows = []
    match_count = 0  # Counter for matched rows
    
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')  # Read TSV file
        header = next(reader)  # Get the header row
        
        for row in reader:
            try:
                p_value = float(row[-1])  # Convert the last column to a float
                if p_value < threshold:
                    filtered_rows.append(row)
                    match_count += 1
            except ValueError:
                # Skip rows where the last column isn't a valid float
                continue
    
    print(f"Number of rows with p-value < {threshold}: {match_count}")
    return header, filtered_rows

# Example usage
file_path = ".../Heart/minP/GWAS_60000_minP.tsv"
header, filtered_rows = filter_rows_by_p_value(file_path)

print("Header:", header)
print("\nRows with P value < 1.953125e-10:")
for row in filtered_rows:
    print(row)
#############################################
############################################# Manhattan Plot
import matplotlib.pyplot as plt    
def plot_manhattan(filepath, significance=-np.log10(5e-8), title=None, save_path=None):
    # Load the data
    df = pd.read_csv(filepath, sep='\t')
    
    # Calculate -log10 of p-values
    df['-log10(P)'] = -np.log10(df['P'])
    
    # Prepare data for plotting
    df['ind'] = range(len(df))
    df_grouped = df.groupby(('CHR'))
    
    # Create the plot
    fig = plt.figure(figsize=(25, 15))
    ax = fig.add_subplot(111)
    colors = ['#0000ff', '#4682b4']
    
    x_labels = []
    x_labels_pos = []
    for num, (name, group) in enumerate(df_grouped):
        group.plot(kind='scatter', x='ind', y='-log10(P)', color=colors[num % len(colors)], ax=ax)
        x_labels.append(name)
        x_labels_pos.append((group['ind'].iloc[-1] + group['ind'].iloc[0]) / 2)
    
    ax.set_xticks(x_labels_pos)
    ax.set_xticklabels(x_labels, rotation=90, fontsize=22)
    ax.tick_params(axis='y', labelsize=22)
    
    # Plot significance line
    plt.axhline(y=significance, color='grey', linestyle='--')
    
    # Increase the size of the axis titles
    ax.set_xlabel('Chromosome', fontsize=20)
    ax.set_ylabel('-log10(p-value)', fontsize=20)
    
    if title:
        plt.title(title, fontsize=24)  # Increase the title size as well if needed
    
    if save_path:
        plt.savefig(save_path, quality=100)
    
    plt.show()

# Use the function to plot
plot_manhattan('.../Heart/minP/GWAS_60000_minP.tsv', significance=-np.log10(5e-8/1))
