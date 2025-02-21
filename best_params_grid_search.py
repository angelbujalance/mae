import os
import pandas as pd

base_folder = "pretrain_mae"

best_loss = float('inf')
best_ncc = 0
best_params = {}
all_params = {}
all_params['loss'] = []
all_params['ncc'] = []
all_params['params'] = []
for subfolder in os.listdir(base_folder):
    subfolder_path = os.path.join(base_folder, subfolder)
    if os.path.isdir(subfolder_path):

        try:
            print(f"Exploring subfolder: {subfolder_path}")
            df = pd.read_json(os.path.join(subfolder_path, "loss_ncc_dict.json"))
            all_params['loss'].append(df['loss'].min())
            all_params['ncc'].append(df['ncc'].max())
            all_params['params'].append(subfolder)

            if df['ncc'].max() > best_ncc:
                best_ncc = df['ncc'].max()
                best_params['ncc'] = best_ncc
                best_params['ncc_best_params'] = subfolder

            if df['loss'].min() < best_loss:
                best_loss = df['loss'].min()
                best_params['loss'] = best_loss
                best_params['loss_best_params'] = subfolder
        except FileNotFoundError:
            continue


print("best grid params:", best_params)

print("10 best combinations of params:")
print(pd.DataFrame.from_dict(all_params).sort_values(by=["loss"]).head(10))