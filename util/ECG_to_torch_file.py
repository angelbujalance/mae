import pandas as pd
import torch
from lxml import etree
import os
from lxml.etree import tostring
import re


initial_file_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(os.path.dirname(initial_file_dir))

# Change the working directory to the parent directory
os.chdir(parent_dir)
print(parent_dir)

# Directory containing the files (update this with your actual directory)
ecg_path = '/gpfs/work2/0/aus20644/data/ukbiobank/ecg/20205_12_lead_ecg_at_rest/imaging_visit_array_0'
os.chdir(ecg_path)

leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
def lead_preprocessing(lead_root:str):
    lead = root.find(f"StripData/WaveformData[@lead='{lead_root}']")

    replace_mark_beg = re.sub(r"^[^\s]*\s[^\s]*\s*" , "", tostring(lead).decode("utf-8").strip())
    replace_mask_end = re.sub(r"\s", "", re.sub(r"[^\d]*$" , "", replace_mark_beg)).split(",")
    ecg_lead_tensor = torch.FloatTensor([int(num) for num in replace_mask_end])

    return ecg_lead_tensor

ECG_tensors = []
for filename in os.listdir(ecg_path):
    if filename.endswith(".xml"):
        file_path = os.path.join(ecg_path, filename)

        parser = etree.XMLParser(huge_tree=True)
        tree = etree.parse(file_path, parser)
        root = tree.getroot()

        try:
            ecg_lead_tensors = [lead_preprocessing(lead) for lead in leads]

            ecg_participant_tensors = torch.stack(ecg_lead_tensors)
            ECG_tensors.append(ecg_participant_tensors)

            # Asserts that the size of the tensor matches 12 ECG-leads and 5000 data samples
            assert ecg_participant_tensors.shape == torch.Size([12, 5000]), (
                    f"Tensor size mismatch: check filename {filename}."
                    )

        except TypeError as te:
            print(f"TypeError: A Type Error ocuured when preprocessing a lead root in filename {filename}.")
            pass


ECG_tensors = torch.stack(ECG_tensors, dim=0)

print(ECG_tensors.shape)

os.chdir(parent_dir + '/mae/data')

torch.save(ECG_tensors, "ECG_leads_full_pretraining.pt")
