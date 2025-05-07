import pandas as pd
import torch
from lxml import etree
import os
from lxml.etree import tostring
import re
from scipy.signal import savgol_filter
from pre_process_ecg_utils import notch_filter, bandpass_filter, leads, highpass_filter


initial_file_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(os.path.dirname(initial_file_dir))

# Change the working directory to the parent directory
os.chdir(parent_dir)
print(parent_dir)

# Directory containing the files (update this with your actual directory)
ecg_path = '/gpfs/work2/0/aus20644/data/ukbiobank/ecg/20205_12_lead_ecg_at_rest/imaging_visit_array_0'
os.chdir(ecg_path)

# leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
def lead_preprocessing(lead_root:str, fs_original=500):
    lead = root.find(f"StripData/WaveformData[@lead='{lead_root}']")

    replace_mark_beg = re.sub(r"^[^\s]*\s[^\s]*\s*" , "", tostring(lead).decode("utf-8").strip())
    replace_mask_end = re.sub(r"\s", "", re.sub(r"[^\d]*$" , "", replace_mark_beg)).split(",")
    ecg_lead_tensor = torch.FloatTensor([int(num) for num in replace_mask_end])

    # Apply pre-processing to the ECG lead https://www.nature.com/articles/s44161-024-00564-3

    # 1. Apply high pass filter
    # low_pass_filtered_lead = bandpass_filter(waveform_values, lowcut=0.5,
    #                                         highcut=100, fs=fs_original)
    high_pass_filtered_lead = highpass_filter(ecg_lead_tensor, cut=0.5,
                                                fs=fs_original)

    # 2. Apply notch filter (60 Hz)
    notch_filter_lead = notch_filter(high_pass_filtered_lead, fs_original, freq=60)
    # find out if frequency = 50 or 60Hz, asking Samuel

    # 3. Apply Savitzky-Golay filter
    pre_processed_lead = savgol_filter(notch_filter_lead, window_length=15, polyorder=3)

    return ecg_lead_tensor, torch.from_numpy(pre_processed_lead)

ECG_tensors = []
ECG_processed = []
ids = []

count = 0
for filename in os.listdir(ecg_path):
    if filename.endswith(".xml"):
        file_path = os.path.join(ecg_path, filename)

        parser = etree.XMLParser(huge_tree=True)
        tree = etree.parse(file_path, parser)
        root = tree.getroot()

        # try:
        ecg_lead_tensors, pre_processed_lead = zip(*[lead_preprocessing(lead) for lead in leads])

        ecg_participant_tensors = torch.stack(ecg_lead_tensors)
        ecg_participant_tensors_processed = torch.stack(pre_processed_lead)
        ECG_tensors.append(ecg_participant_tensors)
        ECG_processed.append(ecg_participant_tensors_processed)
        ids.append(filename.split('_')[0])

        # Asserts that the size of the tensor matches 12 ECG-leads and 5000 data samples
        assert ecg_participant_tensors_processed.shape == torch.Size([12, 5000]), (
                f"Tensor size mismatch: check filename {filename}."
                )

        # except TypeError as te:
        #     print(f"TypeError: A Type Error ocuured when preprocessing a lead root in filename {filename}.")
        #     print(te)
        #     exit()
        #     pass

        count += 1
        if count == 10:
                break


ECG_tensors = torch.stack(ECG_tensors, dim=0)

ECG_processed = torch.stack(ECG_processed, dim=0)

print("Original tensor shape before pre-processing:", ECG_tensors.shape)
print("Pre-processed tensor shape:", ECG_tensors.shape)

os.chdir(parent_dir + '/data')

# torch.save(ECG_tensors, "ECG_leads_full_pretraining.pt")

full_data_ECGs_w_IDs = {"ECG_tensors": ECG_processed, "ECG_originals": ECG_tensors, "IDs": ids}

torch.save(full_data_ECGs_w_IDs, "ECG_leads_full_pretraining_w_IDs_50.pth")