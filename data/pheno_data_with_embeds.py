import pandas as pd
import os
import re
import numpy as np
from lxml import etree
from lxml.etree import tostring
from datetime import date


# Get the directory of the current Python file
initial_file_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = "/gpfs"

# Change the working directory to the parent directory
os.chdir(parent_dir)

print(f"Initial working directory: {initial_file_dir}")
print(f"Current working directory: {os.getcwd()}")

# Load the TSV file (gzip compressed) with the relevant column names
pheno_data = pd.read_csv('work2/0/aus20644/data/ukbiobank/phenotypes/ukb678882.tab.gz',
                       sep='\t', compression='gzip',
                       usecols=['f.eid','f.21003.2.0', 'f.31.0.0', 'f.12340.2.0', 'f.22338.2.0',
                                'f.24100.2.0', 'f.24105.2.0', 'f.24106.2.0', 'f.24140.2.0'],
                       ) # age imaginig visit, sex, QRS duration, QRS num

pheno_data = pheno_data.rename(columns={'f.21003.2.0': 'age_imaging_visit',
                           'f.31.0.0': 'sex',
                           'f.12340.2.0': 'QRS_duration',
                           'f.22338.2.0': 'QRS_num',
                           'f.24100.2.0': 'LV_diast_vol',
                           'f.24105.2.0': 'LV_myoc_mass',
                           'f.24106.2.0': 'RV_diast_vol',
                           'f.24140.2.0': 'LV_myoc_thick'
                           })

pheno_data = pheno_data.drop_duplicates(subset='f.eid')

print(pheno_data.head(10))
print("pheno data loaded and renamed important cols")

os.chdir(initial_file_dir)

embeds_data_train = pd.read_csv('train_ECG_latents_embeds_size_576_CL_3D_data_ALL_pheno2.csv')
embeds_data_train = embeds_data_train.merge(pheno_data, left_on="ID", right_on="f.eid")
embeds_data_train.to_csv("embeds_train_ECG_latents_w_pred_vars_emb_size_576_CL_3D_data_ALL_pheno2.csv", index=False)

del embeds_data_train

embeds_data_val = pd.read_csv('val_ECG_latents_embeds_size_576_CL_3D_data_ALL_pheno2.csv')
embeds_data_val = embeds_data_val.merge(pheno_data, left_on="ID", right_on="f.eid")
embeds_data_val.to_csv("embeds_val_ECG_latents_w_pred_vars_emb_size_576_CL_3D_data_ALL_pheno2.csv", index=False)