# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
import os
import argparse

import json
from typing import Tuple
import numpy as np
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
# from torch.utils.tensorboard import SummaryWriter
import wandb

# assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_

import util.lr_decay as lrd
import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.lars import LARS
from util.callbacks import EarlyStop
from util.dataset import SignalDataset

import models_vit

from engine_finetune import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('MAE linear probing for image classification', add_help=False)
    # Basic parameters
    parser.add_argument('--batch_size', default=512, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch224', type=str, metavar='MODEL',
                        help='Name of model to train')
    
    parser.add_argument('--input_channels', type=int, default=5, metavar='N',
                        help='input channels')
    parser.add_argument('--input_electrodes', type=int, default=65, metavar='N',
                        help='input electrodes')
    parser.add_argument('--time_steps', type=int, default=37000, metavar='N',
                        help='input length')
    parser.add_argument('--input_size', default=(5, 65, 37000), type=Tuple,
                        help='images input size')

    parser.add_argument('--patch_height', type=int, default=65, metavar='N',
                        help='patch height')
    parser.add_argument('--patch_width', type=int, default=200, metavar='N',
                        help='patch width')
    parser.add_argument('--patch_size', default=(65, 200), type=Tuple,
                        help='patch size')
                        
    # Augmentation parameters
    parser.add_argument('--masking_blockwise', action='store_true', default=False,
                        help='Masking blockwise in channel and time dimension (instead of random masking)')
    parser.add_argument('--mask_ratio', default=0.0, type=float,
                        help='Masking ratio (percentage of removed patches)')
    parser.add_argument('--mask_c_ratio', default=0.0, type=float,
                        help='Masking ratio in channel dimension (percentage of removed patches)')
    parser.add_argument('--mask_t_ratio', default=0.0, type=float,
                        help='Masking ratio in time dimension (percentage of removed patches)')
    
    parser.add_argument('--jitter_sigma', default=0.2, type=float,
                        help='Jitter sigma N(0, sigma) (default: 0.2)')
    parser.add_argument('--rescaling_sigma', default=0.5, type=float,
                        help='Rescaling sigma N(0, sigma) (default: 0.5)')
    parser.add_argument('--ft_surr_phase_noise', default=0.075, type=float,
                        help='Phase noise magnitude (default: 0.075)')
    parser.add_argument('--freq_shift_delta', default=0.005, type=float,
                        help='Delta for the frequency shift (default: 0.005)')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0 for linear probe following MoCo v1)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=0.1, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    # parser.add_argument('--layer_decay', type=float, default=0.75,
    #                     help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # Callback parameters
    parser.add_argument('--patience', default=-1, type=float,
                        help='Early stopping whether val is worse than train for specified nb of epochs (default: -1, i.e. no early stopping)')
    parser.add_argument('--max_delta', default=0, type=float,
                        help='Early stopping threshold (val has to be worse than (train+delta)) (default: 0)')

    # Criterion parameters
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true', default=False)
    parser.add_argument('--attention_pool', action='store_true', default=False)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--downstream_task', default='classification', type=str,
                        help='downstream task (default: classification)')

    parser.add_argument('--data_path', default='', type=str,
                        help='dataset path (default: None)')
    parser.add_argument('--labels_path', default='', type=str,
                        help='labels path (default: None)')
    parser.add_argument('--labels_mask_path', default='', type=str,
                        help='labels path (default: None)')

    parser.add_argument('--val_data_path', default='', type=str,
                        help='validation dataset path (default: None)')
    parser.add_argument('--val_labels_path', default='', type=str,
                        help='validation labels path (default: None)')
    parser.add_argument('--val_labels_mask_path', default='', type=str,
                        help='validation labels path (default: None)')

    parser.add_argument('--lower_bnd', type=int, default=0, metavar='N',
                        help='lower_bnd')
    parser.add_argument('--upper_bnd', type=int, default=0, metavar='N',
                        help='upper_bnd')
    
    parser.add_argument('--nb_classes', default=2, type=int,
                        help='number of the classification types')
    parser.add_argument('--pos_label', default=0, type=int,
                        help='classification type with the smallest count')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='',
                        help='path where to tensorboard log (default: ./logs)')
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--wandb_project', default='',
                        help='project where to wandb log')
    parser.add_argument('--wandb_id', default='', type=str,
                        help='id of the current run')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    
    parser.add_argument('--plot_attention_map', action='store_true', default=False)
    parser.add_argument('--plot_embeddings', action='store_true', default=False)
    parser.add_argument('--save_embeddings', action='store_true', default=False,
                        help='save model embeddings')
    parser.add_argument('--save_logits', action='store_true', default=False,
                        help='save model logits')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=32, type=int)
    parser.add_argument('--pin_mem', action='store_true', default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

def main(args):
    args.input_size = (args.input_channels, args.input_electrodes, args.time_steps)
    args.patch_size = (args.patch_height, args.patch_width)

    # misc.init_distributed_mode(args)
    args.distributed = False

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    
    dataset_train = SignalDataset(data_path=args.data_path, labels_path=args.labels_path, labels_mask_path=args.labels_mask_path,
                                  downstream_task=args.downstream_task, train=True, args=args)
    dataset_val = SignalDataset(data_path=args.val_data_path, labels_path=args.val_labels_path, labels_mask_path=args.val_labels_mask_path, 
                                downstream_task=args.downstream_task, train=False, args=args)

    # train balanced
    class_weights = 2.0 / (2.0 * torch.Tensor([1.0, 1.0])) # total_nb_samples / (nb_classes * samples_per_class)

    print("Training set size: ", len(dataset_train))
    print("Validation set size: ", len(dataset_val))

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # tensorboard logging
    if False: #global_rank == 0 and args.log_dir and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    elif False: #args.log_dir and args.eval and "checkpoint" not in args.resume.split("/")[-1]:
        log_writer = SummaryWriter(log_dir=args.log_dir + "/eval")
    else:
        log_writer = None

    # wandb logging
    if args.wandb == True:
        config = vars(args)
        if args.wandb_id:
            wandb.init(project=args.wandb_project, id=args.wandb_id, config=config, entity="oturgut")
        else:
            wandb.init(project=args.wandb_project, config=config, entity="oturgut")

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        sampler=sampler_train,
        # shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, 
        sampler=sampler_val,
        # shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    model = models_vit.__dict__[args.model](
        img_size=args.input_size,
        patch_size=args.patch_size,
        num_classes=args.nb_classes,
        global_pool=args.global_pool,
        attention_pool=args.attention_pool,
        masking_blockwise=args.masking_blockwise,
        mask_ratio=args.mask_ratio,
        mask_c_ratio=args.mask_c_ratio,
        mask_t_ratio=args.mask_t_ratio
    )

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        checkpoint_input_size = checkpoint['args'].input_size
        interpolate_pos_embed(model, checkpoint_model, checkpoint_input_size)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        if args.global_pool:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        elif args.attention_pool:
            pass
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer: following MoCo v3
        trunc_normal_(model.head.weight, std=0.01)

    # for linear prob only
    # hack: revise model's head with BN
    model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
    # freeze all but the head
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.head.named_parameters():
        p.requires_grad = True

    model.to(device, non_blocking=True)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('Number of params (K): %.2f' % (n_parameters / 1.e3))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 4

    print("base lr: %.2e" % (args.lr * 4 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    # param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
    #     no_weight_decay_list=model_without_ddp.no_weight_decay(),
    #     layer_decay=args.layer_decay
    # )
    # optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    optimizer = LARS(model_without_ddp.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(optimizer)
    loss_scaler = NativeScaler()

    class_weights = class_weights.to(device=device, non_blocking=True)
    if args.downstream_task == 'regression':
        criterion = torch.nn.MSELoss()
    elif args.smoothing > 0.:
        # LabelSmoothingCrossEntropy(smoothing=args.smoothing) (see timm.loss)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    print("criterion = %s" % str(criterion))

    if not args.eval:
        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        sub_strings = args.resume.split("/")
        if "checkpoint" in sub_strings[-1]:
            nb_ckpts = 1
        else:
            nb_ckpts = int(sub_strings[-1])+1

        for epoch in range(0, nb_ckpts):
            if "checkpoint" not in sub_strings[-1]:
                args.resume = "/".join(sub_strings[:-1]) + "/checkpoint-" + str(epoch) + ".pth"

            misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

            test_stats, test_history = evaluate(data_loader_val, model, device, epoch, log_writer=log_writer, args=args)
            if args.downstream_task == 'classification':
                print(f"Accuracy / F1 / AUROC / AUPRC of the network on the {len(dataset_val)} test images: {test_stats['acc']:.2f}% /"
                      f"{test_stats['f1']:.2f}% / {test_stats['auroc']:.2f}% / {test_stats['auprc']:.2f}%")
            elif args.downstream_task == 'regression':
                print(f"Root Mean Squared Error (RMSE) / Mean Absolute Error (MAE) / Pearson Correlation Coefficient (PCC) / R Squared (R2)",
                      f"of the network on the {len(dataset_val)} test images: {test_stats['rmse']:.4f} / {test_stats['mae']:.4f} /",
                      f"{test_stats['pcc']:.4f} / {test_stats['r2']:.4f}")

            if args.wandb:
                wandb.log(test_history)
            
        exit(0)

    # Define callbacks
    early_stop = EarlyStop(patience=args.patience, max_delta=args.max_delta)
    
    print(f"Start training for {args.epochs} epochs")

    if args.downstream_task == 'classification':
        eval_criterion = "auroc"
    elif args.downstream_task == 'regression':
        eval_criterion = "pcc"

    best_stats = {'loss':np.inf, 'acc':0.0, 'f1':0.0, 'auroc':0.0, 'auprc':0.0, 'rmse':np.inf, 'mae':np.inf, 'pcc':0.0, 'r2':-1.0}
    best_eval_scores = {'count':0, 'nb_ckpts_max':5, 'eval_criterion':[best_stats[eval_criterion]]}
    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()

        if True: #args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats, train_history = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, 
                                                     loss_scaler, max_norm=None, log_writer=log_writer, args=args)

        test_stats, test_history = evaluate(data_loader_val, model, device, epoch, log_writer=log_writer, args=args)

        if eval_criterion == "loss" or eval_criterion == "rmse" or eval_criterion == "mae":
            if early_stop.evaluate_decreasing_metric(val_metric=test_stats[eval_criterion]):
                break
            if args.output_dir and test_stats[eval_criterion] <= max(best_eval_scores['eval_criterion']):
                # save the best 5 (nb_ckpts_max) checkpoints, even if they appear after the best checkpoint wrt time
                if best_eval_scores['count'] < best_eval_scores['nb_ckpts_max']:
                    best_eval_scores['count'] += 1
                else:
                    best_eval_scores['eval_criterion'] = sorted(best_eval_scores['eval_criterion'])
                    best_eval_scores['eval_criterion'].pop()
                best_eval_scores['eval_criterion'].append(test_stats[eval_criterion])

                misc.save_best_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, test_stats=test_stats, evaluation_criterion=eval_criterion, 
                    mode="decreasing")
        else:
            if early_stop.evaluate_increasing_metric(val_metric=test_stats[eval_criterion]):
                break
            if args.output_dir and test_stats[eval_criterion] >= min(best_eval_scores['eval_criterion']):
                # save the best 5 (nb_ckpts_max) checkpoints, even if they appear after the best checkpoint wrt time
                if best_eval_scores['count'] < best_eval_scores['nb_ckpts_max']:
                    best_eval_scores['count'] += 1
                else:
                    best_eval_scores['eval_criterion'] = sorted(best_eval_scores['eval_criterion'], reverse=True)
                    best_eval_scores['eval_criterion'].pop()
                best_eval_scores['eval_criterion'].append(test_stats[eval_criterion])

                misc.save_best_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, test_stats=test_stats, evaluation_criterion=eval_criterion, 
                    mode="increasing")

        best_stats['loss'] = min(best_stats['loss'], test_stats['loss'])
        
        if args.downstream_task == 'classification':
            # update best stats
            best_stats['f1'] = max(best_stats['f1'], test_stats['f1'])
            best_stats['acc'] = max(best_stats['acc'], test_stats["acc"])
            best_stats['auroc'] = max(best_stats['auroc'], test_stats['auroc'])
            best_stats['auprc'] = max(best_stats['auprc'], test_stats['auprc'])

            print(f"Accuracy / F1 / AUROC / AUPRC of the network on the {len(dataset_val)} test images: {test_stats['acc']:.1f}% /",
                  f"{test_stats['f1']:.1f}% / {test_stats['auroc']:.1f}% / {test_stats['auprc']:.1f}%")
            print(f'Max Accuracy / F1 / AUROC / AUPRC: {best_stats["acc"]:.2f}% / {best_stats["f1"]:.2f}% /',
                  f'{best_stats["auroc"]:.2f}% / {best_stats["auprc"]:.2f}%\n')

        elif args.downstream_task == 'regression':
            # update best stats
            best_stats['rmse'] = min(best_stats['rmse'], test_stats['rmse'])
            best_stats['mae'] = min(best_stats['mae'], test_stats['mae'])
            best_stats['pcc'] = max(best_stats['pcc'], test_stats['pcc'])
            best_stats['r2'] = max(best_stats['r2'], test_stats['r2'])

            print(f"Root Mean Squared Error (RMSE) / Mean Absolute Error (MAE) / Pearson Correlation Coefficient (PCC) / R Squared (R2)",
                  f"of the network on the {len(dataset_val)} test images: {test_stats['rmse']:.4f} / {test_stats['mae']:.4f} /",
                  f"{test_stats['pcc']:.4f} / {test_stats['r2']:.4f}")
            print(f'Min Root Mean Squared Error (RMSE) / Min Mean Absolute Error (MAE) / Max Pearson Correlation Coefficient (PCC) /',
                  f'Max R Squared (R2): {best_stats["rmse"]:.4f} / {best_stats["mae"]:.4f} / {best_stats["pcc"]:.4f} / {best_stats["r2"]:.4f}\n')

        log_stats = {**{f'train_{k}': str(v) for k, v in train_stats.items()},
                        **{f'test_{k}': str(v) for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        if args.wandb:
            wandb.log(train_history | test_history | {"Time per epoch [sec]": total_time})

    if args.wandb:
        wandb.log({f'Best Statistics/{k}': v for k, v in best_stats.items()})


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)