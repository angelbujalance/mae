import torch
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

sys.path.append(os.path.abspath(".."))
import models_mae
import pandas as pd

torch.manual_seed(42)

model_name = 'mae_vit_mediumDeep_patchX' # 'mae_vit_smallDeep_patchX' # 'mae_vit_tinyDeep_patchX' # also mae_vit_smallDeep_patchX, mae_vit_mediumDeep_patchX, mae_vit_base
chkpt_model = '/home/abujalancegome/deep_risk/mae/MAE_pretrain/base/checkpoint-129-loss-0.1812.pth' # best base
chkpt_model = '/home/abujalancegome/deep_risk/mae/MAE_pretrain/checkpoint-141-loss-0.1758.pth' # model after CL
chkpt_model = '/home/abujalancegome/deep_risk/ECG-CMR-CL/CL_results/medium3D/ECGEncoder_checkpoint-9-loss-3.60972042289781.pth' # after CL w/ 3D CMR
chkpt_model = '/home/abujalancegome/deep_risk/ECG-CMR-CL/CL_results/medium3D_all_labels/ECGEncoder_checkpoint-12-loss-3.4776823461791615.pth' # after CL w/ 3D CMR all phenos
chkpt_model = '/home/abujalancegome/deep_risk/ECG-CMR-CL/CL_results/medium3D_exp_all_labels/ECGEncoder_checkpoint-12-loss-3.358973595831129.pth'  # after CL w/ 3D CMR all phenos (2)

model_mae = models_mae.__dict__[model_name](
    img_size=(12, 5000),
    patch_size=(1, 100),
    norm_pix_loss=False,
    ncc_weight=0.1
)

print('Model loaded.')

checkpoint = torch.load(chkpt_model, map_location='cpu', weights_only=False)

try:
    model_mae.load_state_dict(checkpoint['model'], strict=False)
except:
    model_mae.load_state_dict(checkpoint['model_state_dict'], strict=False)


# Data and ID paths
data_paths = [
    "/home/abujalancegome/deep_risk/data/ECG_leads_train.pt",
    "/home/abujalancegome/deep_risk/data/ECG_leads_val.pt",
]

id_paths = [
    "/home/abujalancegome/deep_risk/data/ECG_ids_train.pt",
    "/home/abujalancegome/deep_risk/data/ECG_ids_val.pt",
]

batch_size = 50
all_latents = []
all_ids = []

# Loop through all datasets
for data_path, id_path in zip(data_paths, id_paths):
    path_name = data_path.split("_")[-1].split(".")[0]
    print(path_name)

    # Load tensors and IDs
    if path_name == "train":
        full_tensor = torch.load(data_path)[:12000, :, :]
    else:
        full_tensor = torch.load(data_path)[:4000, :, :]
    ids = torch.load(id_path)

    # Create DataLoader
    dataset = TensorDataset(full_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Process in batches
    for i, batch in enumerate(loader):
        batch_tensor = batch[0].float()
        latent = model_mae(batch_tensor, mask_ratio=0.7, return_latent=True)
        z1 = latent[:, 1:, ...].mean(dim=1)     # global average pooling
        all_latents.append(z1.detach().cpu())

        # Add corresponding IDs
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(ids))
        all_ids.extend(ids[start_idx:end_idx])

    # Stack latents and convert IDs
    final_latents = torch.cat(all_latents, dim=0).numpy()

    print(f"Embeddings of size: {final_latents.shape[1]}")
    final_ids = [str(id_.item()) for id_ in all_ids]

    # Create DataFrame and save
    df = pd.DataFrame(final_latents)
    df.insert(0, "ID", final_ids)
    df.to_csv(os.path.join("data", f"{path_name}_ECG_latents_embeds_size_{final_latents.shape[1]}_CL_3D_data_ALL_pheno2.csv"), index=False)