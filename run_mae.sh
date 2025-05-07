#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=MAE_ECG
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=40:00:00
#SBATCH --output=output/mae_main_ECG_output.out

# Activate your environment
source activate mae3
# pip install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0
pip install timm==0.4.12
pip install numpy==1.26.4
pip install scipy==1.11.3
pip install scikit-learn==1.3.2
pip install umap-learn==0.5.5
pip install seaborn==0.13.0
pip install sentry-sdk==1.35.0
pip install tqdm==4.66.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
# Directory containing ECG samples
# export IMAGENET_DIR=/gpfs/work2/0/aus20644/data/ukbiobank/ecg/20205_12_lead_ecg_at_rest/imaging_visit_array_0/2378863_20205_2_0.xml

DATA_DIR="$HOME/deep_risk/data/ECG_leads_train.pt"
VAL_DIR="$HOME/deep_risk/data/ECG_leads_val.pt"
patch_width=(100)

export CUDA_LAUNCH_BLOCKING=1

learning_rates=(1e-3 1e-4) # 1e-4 already finished 5e-4 1e-5 1e-6
weight_decays=(1e-3 1e-4 1e-6)
mask_ratios=(0.7 0.75 0.8)

learning_rates=(1e-4) # 1e-4 already finished 5e-4 1e-5 1e-6
weight_decays=(1e-4)
mask_ratios=(0.7)

# model mae_vit_base = 768 embedding size
# model mae_vit_mediumDeep_patchX = 576 embedding size
# model mae_vit_smallDeep_patchX = 384 embedding size
# model mae_vit_tinyDeep_patchX = 192 embedding size


# Runs MAE for ECG signals
# python main_pretrain.py --eval --resume mae_finetuned_vit_base.pth --model vit_base_patch16 --batch_size 16 --data_path ${IMAGENET_DIR} --val_data_path ${IMAGENET_DIR}

# python main_pretrain.py --resume mae_finetuned_vit_base.pth --model mae_vit_base --batch_size 16 --data_path ${IMAGENET_DIR} --val_data_path ${IMAGENET_DIR}
# Grid search loop
for lr in "${learning_rates[@]}"
do
    for wd in "${weight_decays[@]}"
    do
        for mr in "${mask_ratios[@]}"
        do
            echo "Running with lr=$lr, weight_decay=$wd, mask_ratio=$mr"
            
            python main_pretrain.py --resume '' \
                --model mae_vit_base \
                --batch_size 64 \
                --data_path ${DATA_DIR} \
                --val_data_path ${VAL_DIR} \
                --input_channels 1 \
                --input_electrodes 12 \
                --time_steps 5000 \
                --patch_height 1 \
                --patch_width ${patch_width} \
                --epochs 250 \
                --patience 15 \
                --output_dir MAE_pretrain/base \
                --lr ${lr} \
                --weight_decay ${wd} \
                --mask_ratio ${mr}
        done
    done
done
