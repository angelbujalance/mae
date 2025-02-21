import torch
import os
from dataset import SignalDataset


parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(parent_dir)
os.chdir(parent_dir)

# Step 1: Load the ECG Data
ECG_full_data = torch.load("data/ECG_leads_full_pretraining_test.PT")
# ecg_data = ecg_data[:, None, :, :]  # Add channel dimension (patients, 1, 12, 2500)


#Z-score Normalization
# Step 2: Remove signals that are all zeros
# valid_indices = ~torch.all(ECG_full_data == 0, dim=(1, 2))

# Filter out all-zero participants
# filtered_ECG_data = ECG_full_data[valid_indices]

# print(f"Removed {len(valid_indices) - torch.sum(valid_indices)} all-zero signals.")
# print(f"Remaining signals for training: {len(ecg_data)}")

# Step 3: Normalize the Data
# Calculate mean and standard deviation across participants
mean = torch.mean(ECG_full_data, dim=(0), keepdim=True)  # Shape: (1, leads, samples)
std = torch.std(ECG_full_data, dim=(0), keepdim=True)    # Shape: (1, leads, samples)

# Avoid division by zero (set std to 1 where it's 0)
std[std == 0] = 1.0

# Normalize the data (Z-score normalization)
normalized_ECG_data = (ECG_full_data - mean) / std

print(type(normalized_ECG_data))

print(normalized_ECG_data.shape)

os.chdir(parent_dir + '/data')

torch.save(normalized_ECG_data, "ECG_leads_full_pret_norm.pt")

print(type(torch.load("ECG_leads_full_pret_norm.pt")))

SignalDataset("ECG_leads_full_pret_norm.pt")