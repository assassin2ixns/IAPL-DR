# -*- coding: utf-8 -*-
import os
import datetime
import time
import torch
import random
import numpy as np
import argparse
from pathlib import Path
from utils.misc import get_rank, init_distributed_mode, save_on_master, is_main_process
from utils.dataset import Dataset_Creator, Dataset_Creator_GenImage, Dataset_Creator_Chameleon, Dataset_Creator_Chameleon_SD
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import OneCycleLR
from engine import train_one_epoch, evaluate
from utils.dream_logging import write_diagnosis_summary, write_jsonl, write_startup_sanity
from models import build_model
import pdb
from test_time import testtime_main
from timm.utils import ModelEmaV2
from timm.utils import get_state_dict


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1', 'y'):
        return True
    if v.lower() in ('no', 'false', 'f', '0', 'n'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
        
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    # dataset parameters
    parser.add_argument('--dataset_path', type=str, default='../dataset')
    parser.add_argument('--img_resolution', type=int, default=256)
    parser.add_argument('--crop_resolution', type=int, default=224)
    parser.add_argument('--train_selected_subsets', nargs='+', required=True)
    parser.add_argument('--test_selected_subsets', nargs='+', required=True)
    parser.add_argument('--dataset', type=str, default='UniversalFakeDetect')


    # training parameters
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--evalbatchsize', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--clip_max_norm', type=float, default=0.)
    parser.add_argument('--lr_drop', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--ema', type=str2bool, default=False)

    # adapter
    parser.add_argument('--vit_adapter_list', type=list, default=[3, 7, 11, 15, 19, 23])
    parser.add_argument('--text_adapter_list', type=list, default=[])

    # model
    parser.add_argument('--method', type=str, default='iapl', choices=['iapl', 'dream_cs'])
    parser.add_argument('--model_variant', type=str, default='clip_adapter')
    parser.add_argument('--backbone', type=str, default='CLIP:ViT-L/14')
    parser.add_argument('--clip_path', type=str, default='/Path/to/ViT-L-14.pt')

    # ctx
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--vision_width', type=int, default=1024)
    parser.add_argument('--prompt_depth', type=int, default=9)
    parser.add_argument('--n_ctx', type=int, default=2)
    parser.add_argument('--ctx_init', type=str, default="a photo of a")
    parser.add_argument('--progress_alpha', type=float, default=0.1)
    parser.add_argument('--condition', type=str2bool, default=False)
    parser.add_argument('--gate', type=str2bool, default=False)

    # tta
    parser.add_argument('--tta', type=str2bool, default=False)
    parser.add_argument('--tta_steps', type=int, default=1)
    parser.add_argument('--selection_p', type=float, default=0.2)
    parser.add_argument('--ois', type=str2bool, default=False)

    # DREAM-CS
    parser.add_argument('--dream_num_experts', type=int, default=3)
    parser.add_argument('--dream_expert_names', nargs='+', default=['fine', 'stable', 'consensus'])
    parser.add_argument('--dream_rank', type=int, default=8)
    parser.add_argument('--dream_router_hidden', type=int, default=32)
    parser.add_argument('--dream_delta_clip', type=float, default=1.0)
    parser.add_argument('--dream_residual_scale_init', type=float, default=1.0)
    parser.add_argument('--dream_apply_init_bias', type=float, default=-3.0)
    parser.add_argument('--dream_route_tau', type=float, default=0.5)
    parser.add_argument('--dream_route_margin', type=float, default=0.01)
    parser.add_argument('--dream_clean_safe_margin', type=float, default=0.0)
    parser.add_argument('--dream_num_train_views', type=int, default=2)
    parser.add_argument('--dream_degradation_pool', nargs='+', default=['jpeg90', 'jpeg75', 'jpeg50', 'resize', 'blur', 'quant', 'webp'])
    parser.add_argument('--dream_expert_forward_chunk', type=int, default=0)
    parser.add_argument('--dream_eval_degradation', type=str, default='none',
                        choices=['none', 'jpeg50', 'jpeg75', 'jpeg90', 'resize', 'blur', 'quant', 'webp'])
    parser.add_argument('--dream_eval_multi_degradations', nargs='+', default=[])
    parser.add_argument('--dream_log_router', type=str2bool, default=True)
    parser.add_argument('--dream_disable_router', type=str2bool, default=False)
    parser.add_argument('--dream_disable_robust', type=str2bool, default=False)
    parser.add_argument('--dream_disable_clean_safe', type=str2bool, default=False)
    parser.add_argument('--dream_disable_route_loss', type=str2bool, default=False)
    parser.add_argument('--dream_disable_expert_correction', type=str2bool, default=False)
    parser.add_argument('--dream_disable_expert_aux', type=str2bool, default=False)
    parser.add_argument('--dream_disable_specialization', type=str2bool, default=False)
    parser.add_argument('--dream_anchor_ckpt', type=str, default='')
    parser.add_argument('--dream_freeze_anchor', type=str2bool, default=False)
    parser.add_argument('--dream_warmup_epochs', type=int, default=1)
    parser.add_argument('--dream_router_start_epoch', type=int, default=1)
    parser.add_argument('--dream_apply_train_floor', type=float, default=0.05)
    parser.add_argument('--dream_use_apply_floor', type=str2bool, default=True)
    parser.add_argument('--dream_tta_safe', type=str2bool, default=True)
    parser.add_argument('--dream_tta_agg', type=str, default='mean', choices=['confidence', 'mean', 'trimmed_mean'])
    parser.add_argument('--dream_tta_disagreement_fallback', type=str2bool, default=True)
    parser.add_argument('--dream_tta_disagreement_thresh', type=float, default=0.05)
    parser.add_argument('--dream_fast_mode', type=str, default='off',
                        choices=['off', 'bank_plus_anchor', 'single_bank'])
    parser.add_argument('--dream_fast_readout', type=str, default='delta_from_prompt',
                        choices=['delta_from_prompt', 'prompt_direct', 'cls_plus_prompt'])
    parser.add_argument('--dream_fast_prompt_delta_norm', type=str2bool, default=True)
    parser.add_argument('--dream_fast_prompt_delta_scale_init', type=float, default=0.1)
    parser.add_argument('--dream_fast_deep_bank', type=str2bool, default=True)
    parser.add_argument('--dream_fast_deep_residual', type=str2bool, default=True)
    parser.add_argument('--dream_fast_return_prompt_tokens', type=str2bool, default=True)
    parser.add_argument('--dream_fast_compare_batchfold', type=str2bool, default=False)
    parser.add_argument('--dream_fast_log_shapes', type=str2bool, default=True)
    parser.add_argument('--dream_fast_detach_anchor_prompt_pool', type=str2bool, default=False)
    parser.add_argument('--dream_fast_force_bank_plus_anchor_for_train', type=str2bool, default=False)
    parser.add_argument('--dream_warmup_freeze_router', type=str2bool, default=True)
    parser.add_argument('--dream_warmup_fixed_apply', type=float, default=0.2)
    parser.add_argument('--dream_balanced_degradation_views', type=str2bool, default=True)
    parser.add_argument('--dream_light_degradation_pool', nargs='+', default=['jpeg90', 'jpeg75'])
    parser.add_argument('--dream_strong_degradation_pool', nargs='+', default=['jpeg50', 'resize', 'blur', 'quant', 'webp'])
    parser.add_argument('--dream_cavr_eps', type=float, default=1e-4)
    parser.add_argument('--dream_save_pred_csv', type=str2bool, default=False)

    # loss
    parser.add_argument('--loss_adapter', type=float, default=1.0)
    parser.add_argument('--loss_contrast', type=float, default=1.0)
    parser.add_argument('--loss_condition', type=float, default=1.0)
    parser.add_argument('--use_contrast', type=str2bool, default=False)
    parser.add_argument('--smooth', type=str2bool, default=False)
    parser.add_argument('--loss_dream_clean', type=float, default=1.0)
    parser.add_argument('--loss_dream_anchor', type=float, default=1.0)
    parser.add_argument('--loss_dream_expert', type=float, default=0.2)
    parser.add_argument('--loss_dream_specialize', type=float, default=0.2)
    parser.add_argument('--loss_dream_anchor_rob', type=float, default=0.2)
    parser.add_argument('--loss_dream_rob', type=float, default=0.5)
    parser.add_argument('--loss_dream_inv', type=float, default=0.05)
    parser.add_argument('--loss_dream_route', type=float, default=0.1)
    parser.add_argument('--loss_dream_apply', type=float, default=0.05)
    parser.add_argument('--loss_dream_clean_safe', type=float, default=0.5)
    parser.add_argument('--loss_dream_res', type=float, default=0.01)
    parser.add_argument('--loss_dream_div', type=float, default=0.005)

    # output
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--pretrained_model', type=str, default="")
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--save_checkpoint_interval', default=30, type=int)
    parser.add_argument('--model_name', type=str, default='CLIP_adapter')
    parser.add_argument('--print_freq', default=50, type=int)
    parser.add_argument('--eval_interval', default=1, type=int)
    parser.add_argument('--eval_first_epoch', type=str2bool, default=True)
    parser.add_argument('--full_eval_last_epoch', type=str2bool, default=True)
    parser.add_argument('--eval_max_batches_per_domain', default=-1, type=int)
    parser.add_argument('--eval_selected_subsets_dev', nargs='+', default=[])
    parser.add_argument('--amp', type=str2bool, default=False)
    parser.add_argument('--amp_dtype', type=str, default='bf16', choices=['fp16', 'bf16'])
    parser.add_argument('--tf32', type=str2bool, default=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local-rank", type=int, help='local rank for DistributedDataParallel')

    return parser

def main(args):
    init_distributed_mode(args)
    if getattr(args, 'method', 'iapl') == 'dream_cs' and getattr(args, 'dream_anchor_ckpt', ''):
        print('WARNING: dream_anchor_ckpt is a plugin/warmstart ablation, not the DREAM-CS Standalone main setting.')
    if getattr(args, 'tf32', True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    if is_main_process() and getattr(args, 'method', 'iapl') == 'dream_cs':
        mode = getattr(args, 'dream_fast_mode', 'off')
        per_view = {'off': args.dream_num_experts + 1, 'bank_plus_anchor': 2, 'single_bank': 1}[mode]
        if getattr(args, 'dream_fast_force_bank_plus_anchor_for_train', False) and mode == 'single_bank':
            print('DREAM-CS Fast: train forward will use bank_plus_anchor because dream_fast_force_bank_plus_anchor_for_train=True.')
        print('DREAM-CS Standalone: label convention real=0 fake=1; TTA={}'.format(args.tta))
        print('DREAM-CS Fast config: mode={} readout={} deep_bank={} deep_residual={} expected_encoder_multiplier_per_view={}'.format(
            mode,
            getattr(args, 'dream_fast_readout', 'delta_from_prompt'),
            getattr(args, 'dream_fast_deep_bank', True),
            getattr(args, 'dream_fast_deep_residual', True),
            per_view,
        ))

    if args.output_dir:
        if os.path.basename(os.path.normpath(args.output_dir)) != args.model_name:
            args.output_dir = os.path.join(args.output_dir, args.model_name)
        os.makedirs(args.output_dir, exist_ok=True)
    output_dir = Path(args.output_dir)

    # set device
    device = torch.device(args.device)

    # fix the seed
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)

    # inter tta step
    if args.tta == True:
        assert args.eval == True
        testtime_main(args)
        exit()

    # infer dataset
    assert args.dataset in ['UniversalFakeDetect', 'GenImage', 'Chameleon', 'Chameleon_SD']
    if args.dataset == 'UniversalFakeDetect':
        dataset_creator = Dataset_Creator(dataset_path=args.dataset_path, batch_size=args.evalbatchsize, num_workers=args.num_workers, img_resolution=args.img_resolution, crop_resolution=args.crop_resolution)
    elif args.dataset == 'GenImage':
        dataset_creator = Dataset_Creator_GenImage(dataset_path=args.dataset_path, batch_size=args.evalbatchsize, num_workers=args.num_workers, img_resolution=args.img_resolution, crop_resolution=args.crop_resolution)               
    elif args.dataset == 'Chameleon':
        dataset_creator = Dataset_Creator_Chameleon(dataset_path=args.dataset_path, batch_size=args.evalbatchsize, num_workers=args.num_workers, img_resolution=args.img_resolution, crop_resolution=args.crop_resolution)
    elif args.dataset == 'Chameleon_SD':
        dataset_creator = Dataset_Creator_Chameleon_SD(dataset_path=args.dataset_path, batch_size=args.evalbatchsize, num_workers=args.num_workers, img_resolution=args.img_resolution, crop_resolution=args.crop_resolution)
                    
    dataset_vals, selected_subsets = dataset_creator.build_dataset("test", selected_subsets=args.test_selected_subsets)
    data_loader_vals = {}
    num_testimage = 0
    for dataset_val, selected_subset in zip(dataset_vals, selected_subsets):
        num_testimage += len(dataset_val)
        if args.distributed:
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_val = SequentialSampler(dataset_val)
        data_loader_vals[selected_subset] = DataLoader(dataset_val, args.evalbatchsize, sampler=sampler_val, drop_last=False, num_workers=args.num_workers)

    data_loader_vals_dev = None
    if len(getattr(args, 'eval_selected_subsets_dev', [])) > 0:
        dataset_vals_dev, selected_subsets_dev = dataset_creator.build_dataset("test", selected_subsets=args.eval_selected_subsets_dev)
        data_loader_vals_dev = {}
        for dataset_val, selected_subset in zip(dataset_vals_dev, selected_subsets_dev):
            if args.distributed:
                sampler_val = DistributedSampler(dataset_val, shuffle=False)
            else:
                sampler_val = SequentialSampler(dataset_val)
            data_loader_vals_dev[selected_subset] = DataLoader(
                dataset_val,
                args.evalbatchsize,
                sampler=sampler_val,
                drop_last=False,
                num_workers=args.num_workers,
            )

    model = build_model(args)
    model = model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    if args.ema:
        model_ema = ModelEmaV2(model_without_ddp, decay=0.9999)
        print('-----------use EMA train mode----------')
    else:
        model_ema = None

    write_startup_sanity(args, model_without_ddp)
    
    if args.eval:
        checkpoint = torch.load(args.pretrained_model, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
        args.current_epoch = checkpoint.get('epoch', 'eval')

        if args.ema and ('model_ema' in checkpoint.keys()):
            model_ema.module.load_state_dict(checkpoint['model_ema'])

        if args.ema and ('model_ema' in checkpoint.keys()):
            evaluate(model_ema.module, data_loader_vals, device, args=args)
        else:
            evaluate(model, data_loader_vals, device, args=args)
        write_diagnosis_summary(args.output_dir, args)
        exit()
    
    # DATA LOADER
    dataset_train = dataset_creator.build_dataset("train", selected_subsets=args.train_selected_subsets)
    
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
    else:
        sampler_train = RandomSampler(dataset_train)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batchsize, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, num_workers=args.num_workers)
    
    optimizer = torch.optim.Adam([{"params": [p for n, p in model.named_parameters() if p.requires_grad]}], lr=args.lr, betas=(args.beta1, 0.999), weight_decay=args.weight_decay)
    
    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=args.gamma)
    
    if args.resume:
        checkpoint = torch.load(args.pretrained_model, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])             

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1

    best_ap = 0
    best_acc = 0
    for epoch in range(args.start_epoch, args.epoch):
        epoch_start_time = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_one_epoch(model, data_loader_train, optimizer, device, epoch, lr_scheduler, args.clip_max_norm, args=args, model_ema = model_ema)
        
        lr_scheduler.step()
        
        if args.output_dir:
            checkpoint_paths = [
                output_dir / 'checkpoint.pth',
                output_dir / 'checkpoint_last.pth',
                output_dir / f'checkpoint_epoch_{epoch}.pth',
            ]

            if (epoch + 1) % args.save_checkpoint_interval == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                weights = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'args': args,
                }
                if model_ema:
                    weights['model_ema'] = get_state_dict(model_ema)
                save_on_master(weights, checkpoint_path)
        
        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        print('Epoch training time {}'.format(epoch_time_str))
        
        args.current_epoch = epoch
        last_epoch = epoch == args.epoch - 1
        eval_due = ((epoch + 1) % max(int(args.eval_interval), 1) == 0)
        if epoch == args.start_epoch and args.eval_first_epoch:
            eval_due = True
        if last_epoch and args.full_eval_last_epoch:
            eval_due = True
        if not eval_due:
            if is_main_process():
                print('Skip evaluation at epoch {} because eval_interval={}'.format(epoch, args.eval_interval))
            continue

        eval_loaders = data_loader_vals
        if data_loader_vals_dev is not None and not (last_epoch and args.full_eval_last_epoch):
            eval_loaders = data_loader_vals_dev
        if args.ema:
            output_strs, cur_ap, cur_acc = evaluate(model_ema.module, eval_loaders, device, args=args)
        else:
            output_strs, cur_ap, cur_acc = evaluate(model, eval_loaders, device, args=args)


        if args.output_dir and is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(f"Epoch {epoch}:" + output_strs + "\n")

            if cur_ap > best_ap:
                best_ap = cur_ap
                checkpoint_path = output_dir / f'checkpoint_best_ap.pth'
                weights = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'args': args,
                }
                if model_ema:
                    weights['model_ema'] = get_state_dict(model_ema)
                save_on_master(weights, checkpoint_path)
                write_jsonl(args.output_dir, 'warning_flags.jsonl', {
                    'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
                    'epoch': epoch,
                    'iter': None,
                    'global_step': None,
                    'split': 'checkpoint',
                    'domain': 'mean',
                    'degradation': getattr(args, 'dream_eval_degradation', 'none'),
                    'seed': getattr(args, 'seed', None),
                    'model_name': getattr(args, 'model_name', None),
                    'lr': optimizer.param_groups[0]['lr'],
                    'batch_size': None,
                    'warning': 'test_domain_checkpoint_selection',
                    'message': 'WARNING: checkpoint_best_ap uses evaluation domains and should not be used for final paper selection.',
                    'metric': 'ap',
                    'value': cur_ap,
                })
            
            if cur_acc > best_acc:
                best_acc = cur_acc
                checkpoint_path = output_dir / f'checkpoint_best_acc.pth'
                weights = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'args': args,
                }
                if model_ema:
                    weights['model_ema'] = get_state_dict(model_ema)
                save_on_master(weights, checkpoint_path)
                write_jsonl(args.output_dir, 'warning_flags.jsonl', {
                    'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
                    'epoch': epoch,
                    'iter': None,
                    'global_step': None,
                    'split': 'checkpoint',
                    'domain': 'mean',
                    'degradation': getattr(args, 'dream_eval_degradation', 'none'),
                    'seed': getattr(args, 'seed', None),
                    'model_name': getattr(args, 'model_name', None),
                    'lr': optimizer.param_groups[0]['lr'],
                    'batch_size': None,
                    'warning': 'test_domain_checkpoint_selection',
                    'message': 'WARNING: checkpoint_best_acc uses evaluation domains and should not be used for final paper selection.',
                    'metric': 'acc',
                    'value': cur_acc,
                })

    write_diagnosis_summary(args.output_dir, args)


if __name__ == "__main__":    
    parser = get_args_parser()
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
