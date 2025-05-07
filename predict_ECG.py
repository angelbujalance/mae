import sys
import os
import requests
import argparse
from sklearn.metrics import mean_squared_error

import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from data.pre_process_ecg_utils import leads

sys.path.append(os.path.abspath(".."))
import models_mae

torch.manual_seed(42)

chkpt_model = '/home/abujalancegome/deep_risk/mae/MAE_pretrain/tiny/checkpoint-136-loss-0.1797.pth' # best tiny e.d. = 192

model_name = 'mae_vit_tinyDeep_patchX' # also mae_vit_smallDeep_patchX, mae_vit_mediumDeep_patchX, mae_vit_base

model_mae = models_mae.__dict__[model_name](
    img_size=(12, 5000),
    patch_size=(1, 100),
    norm_pix_loss=False,
    ncc_weight=0.1
)
print('Model loaded.')

checkpoint = torch.load(chkpt_model, map_location='cpu', weights_only=False)
msg = model_mae.load_state_dict(checkpoint['model'], strict=False)
print(msg)

tensor_path = "/home/abujalancegome/deep_risk/data/ECG_leads_test.pt"

tensor = torch.load(tensor_path)[:, :, :, :] # 500
print(tensor.shape)

loss, pred, mask, imgs = model_mae(tensor.float(), mask_ratio=.7, visualize=True)

print("unpatchifying the predictions...")
pred = model_mae.unpatchify(pred)

ECG_preds = pred[:,:,:,:].squeeze(1).detach().numpy()
print(ECG_preds.shape)
ECG_imgs = imgs[:,0,:,:].detach().numpy()
print(ECG_imgs.shape)

num_leads = 12

coefs = []
for lead in range(num_leads):
    correlation_coefficient = np.corrcoef(ECG_preds[0, lead,:], ECG_imgs[0, lead,:])[0, 1]
    print(f"Cross Correlation Coefficient for lead {leads[lead]}:", np.round(correlation_coefficient,3))
    coefs.append(correlation_coefficient)

print(f"\nCross Correlation Coefficient mean across leads:", np.round( np.mean(coefs) ,3))

mse_errors = []
for lead in range(num_leads):
    mse_x_lead = mean_squared_error(ECG_preds[0,lead,:20], ECG_imgs[0,lead,:20])
    mse_errors.append(mse_x_lead)
    
    print(f"MSE for lead {leads[lead]}:", np.round(mse_x_lead,3))

print(f"\nMSE mean across leads:", np.round( np.mean(mse_errors) ,3))