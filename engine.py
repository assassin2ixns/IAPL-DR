import math
import sys
import time
from itertools import islice
from typing import Iterable

import numpy as np
import torch
import torch.distributed as dist

import utils.misc as utils
from utils.dream_logging import (
    aggregate_records,
    base_record,
    binary_metrics,
    build_prediction_rows,
    compute_train_record,
    concise_console_record,
    debug_first_batch,
    domain_metrics_from_rows,
    ensure_output_dirs,
    mean_metrics_from_domains,
    warning_flags,
    write_corruption_summary,
    write_jsonl,
    write_prediction_csv,
    write_startup_sanity,
    write_top_cases,
    write_diagnosis_summary,
)

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def extract_logits(outputs):
    if isinstance(outputs, dict):
        return outputs['logits']
    if isinstance(outputs, (list, tuple)):
        return outputs[0]
    return outputs


def extract_logits_flat(outputs):
    return extract_logits(outputs).view(-1)


def _progress_enabled():
    return utils.is_main_process() and tqdm is not None and sys.stderr.isatty()


def _make_progress(iterable, total=None, desc='', leave=True):
    if _progress_enabled():
        return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=leave)
    return iterable


def _progress_postfix(progress, values):
    if not hasattr(progress, 'set_postfix'):
        return
    clean = {}
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            value = value.detach().float().mean().cpu().item()
        if isinstance(value, (float, np.floating)):
            if math.isfinite(float(value)):
                clean[key] = '{:.4g}'.format(float(value))
            else:
                clean[key] = str(value)
        else:
            clean[key] = value
    progress.set_postfix(clean, refresh=False)


def _console_print(message):
    if _progress_enabled():
        tqdm.write(message)
    else:
        print(message)


@torch.no_grad()
def gather_together(data):
    world_size = utils.get_world_size()
    if world_size < 2:
        return [data]
    dist.barrier()
    gather_data = [None for _ in range(world_size)]
    dist.all_gather_object(gather_data, data)
    return gather_data


def _detach_loss_dict(loss_dict):
    return {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}


def _print_train_line(epoch, iteration, record):
    fields = concise_console_record(record)
    msg = ['epoch={}'.format(epoch), 'iter={}'.format(iteration)]
    for key, value in fields.items():
        if isinstance(value, float):
            msg.append('{}={:.5g}'.format(key, value))
        else:
            msg.append('{}={}'.format(key, value))
    _console_print('[TRAIN] ' + ' '.join(msg))


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    epoch: int, lr_scheduler=None, max_norm: float = 0, args=None, model_ema=None):
    model.train()
    model_without_ddp = unwrap_model(model)
    if hasattr(model_without_ddp, 'set_epoch'):
        model_without_ddp.set_epoch(epoch)
    if getattr(args, 'method', 'iapl') == 'dream_cs':
        ensure_output_dirs(args.output_dir)

    print_freq = getattr(args, 'print_freq', 50)
    weight_dict = model_without_ddp.criterion_weight_dict
    epoch_records = []
    header_start = time.time()
    end = time.time()
    amp_enabled = bool(getattr(args, 'amp', False)) and device.type == 'cuda'
    amp_dtype = torch.bfloat16 if getattr(args, 'amp_dtype', 'bf16') == 'bf16' else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and getattr(args, 'amp_dtype', 'bf16') == 'fp16')

    try:
        train_total = len(data_loader)
    except TypeError:
        train_total = None
    progress = _make_progress(data_loader, total=train_total, desc='Train epoch {}'.format(epoch), leave=True)

    for iteration, samples in enumerate(progress):
        data_time = time.time() - end
        batch_start = time.time()
        images, labels = [sample.to(device, non_blocking=True) for sample in samples]

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            outputs = model(images)
            loss_dict = model_without_ddp.get_criterion(outputs, labels)
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_dict_reduced = utils.reduce_dict(_detach_loss_dict(loss_dict))
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler.is_enabled():
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
        else:
            losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        batch_time = time.time() - batch_start
        is_dream = isinstance(outputs, dict) and getattr(args, 'method', 'iapl') == 'dream_cs'
        if is_dream and getattr(args, 'dream_log_router', True):
            global_step = epoch * len(data_loader) + iteration
            record = compute_train_record(
                args=args,
                outputs=outputs,
                labels=labels,
                loss_dict=loss_dict_reduced,
                weight_dict=weight_dict,
                model=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
                iteration=iteration,
                global_step=global_step,
                data_time=data_time,
                batch_time=batch_time,
            )
            epoch_records.append(record)
            if iteration == 0:
                debug_first_batch(args, outputs, labels, epoch)
            if utils.is_main_process() and (iteration % print_freq == 0):
                write_jsonl(args.output_dir, 'train_iter_metrics.jsonl', record)
                _print_train_line(epoch, iteration, record)
                _progress_postfix(progress, {
                    'loss': record.get('loss_total'),
                    'acc': record.get('train_acc_final'),
                    'racc': record.get('train_racc_final'),
                    'facc': record.get('train_facc_final'),
                    'apply': record.get('apply_eff_mean', record.get('apply_mean')),
                    'qH': record.get('q_entropy_mean'),
                    'corr': record.get('correction_abs_mean'),
                    'cavr': record.get('cavr'),
                    'active': record.get('active_rate'),
                })
        elif utils.is_main_process() and iteration % print_freq == 0:
            _console_print('[TRAIN] epoch={} iter={} loss_total={:.5g} lr={:.5g}'.format(
                epoch, iteration, loss_value, optimizer.param_groups[0]['lr']))
            _progress_postfix(progress, {
                'loss': loss_value,
                'lr': optimizer.param_groups[0]['lr'],
            })

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        if model_ema:
            model_ema.update(model)
        end = time.time()

    if getattr(args, 'method', 'iapl') == 'dream_cs' and getattr(args, 'dream_log_router', True):
        epoch_record = aggregate_records(epoch_records, args, epoch)
        write_jsonl(args.output_dir, 'train_epoch_metrics.jsonl', epoch_record)
        if utils.is_main_process():
            elapsed = time.time() - header_start
            _console_print('[TRAIN-EPOCH] epoch={} elapsed_sec={:.1f} loss_total={:.5g} acc_final={:.4f} apply={:.4f} q_entropy={:.4f}'.format(
                epoch,
                elapsed,
                epoch_record.get('loss_total', float('nan')),
                epoch_record.get('train_acc_final', float('nan')),
                epoch_record.get('apply_eff_mean', float('nan')),
                epoch_record.get('q_entropy_mean', float('nan')),
            ))


def _gather_prediction_rows(rows):
    gathered = gather_together(rows)
    merged = []
    for rank_rows in gathered:
        merged.extend(rank_rows)
    return merged


@torch.no_grad()
def _evaluate_iapl(model, data_loaders, device, args=None, degradation_name='none'):
    model.eval()
    test_dataset, test_ap, test_acc, test_racc, test_facc = [], [], [], [], []
    output_strs = []
    domain_items = list(data_loaders.items())
    domain_progress = _make_progress(domain_items, total=len(domain_items), desc='Eval domains', leave=True)
    for data_name, data_loader in domain_progress:
        _progress_postfix(domain_progress, {'domain': data_name, 'deg': degradation_name})
        y_true, y_pred = [], []
        max_batches = getattr(args, 'eval_max_batches_per_domain', -1)
        try:
            batch_total = len(data_loader)
        except TypeError:
            batch_total = None
        batch_iterable = data_loader
        if max_batches > 0:
            batch_iterable = islice(data_loader, max_batches)
            if batch_total is not None:
                batch_total = min(batch_total, max_batches)
            else:
                batch_total = max_batches
        batch_progress = _make_progress(
            batch_iterable,
            total=batch_total,
            desc='Eval {}:{}'.format(degradation_name, data_name),
            leave=False,
        )
        for batch_idx, samples in enumerate(batch_progress):
            images, labels = [sample.to(device, non_blocking=True) for sample in samples]
            outputs = model(images)
            probs = extract_logits(outputs).sigmoid().flatten()
            y_pred.extend(probs.detach().cpu().tolist())
            y_true.extend(labels.detach().cpu().flatten().tolist())
        merged_true, merged_pred = [], []
        for part in gather_together(y_true):
            merged_true.extend(part)
        for part in gather_together(y_pred):
            merged_pred.extend(part)
        metrics = binary_metrics(merged_true, merged_pred)
        test_dataset.append(data_name)
        test_ap.append(metrics['ap'])
        test_acc.append(metrics['acc'])
        test_racc.append(metrics['racc'])
        test_facc.append(metrics['facc'])
        if utils.is_main_process():
            _progress_postfix(domain_progress, {
                'domain': data_name,
                'acc': metrics['acc'],
                'ap': metrics['ap'],
            })
            _console_print("({}) acc: {:.2f}; ap: {:.2f}; racc: {:.2f}; facc: {:.2f};".format(
                data_name, metrics['acc'] * 100, metrics['ap'] * 100, metrics['racc'] * 100, metrics['facc'] * 100))

    names = test_dataset + ['mean']
    aps = test_ap + [np.nanmean(test_ap)]
    accs = test_acc + [np.nanmean(test_acc)]
    raccs = test_racc + [np.nanmean(test_racc)]
    faccs = test_facc + [np.nanmean(test_facc)]
    for idx, (name, ap, acc, racc, facc) in enumerate(zip(names, aps, accs, raccs, faccs)):
        s = "({} {:10}) acc: {:.2f}; ap: {:.2f}; racc: {:.2f}; facc: {:.2f};".format(
            idx, name, acc * 100, ap * 100, racc * 100, facc * 100)
        output_strs.append(s)
        if utils.is_main_process():
            _console_print(s)
    return '; '.join(output_strs), np.nanmean(test_ap), np.nanmean(test_acc), {}


@torch.no_grad()
def _evaluate_dream_once(model, data_loaders, device, args=None, degradation_name=None):
    model.eval()
    epoch = getattr(args, 'current_epoch', None)
    degradation = degradation_name if degradation_name is not None else getattr(args, 'dream_eval_degradation', 'none')
    old_deg = getattr(args, 'dream_eval_degradation', 'none')
    if degradation_name is not None:
        args.dream_eval_degradation = degradation_name
    ensure_output_dirs(args.output_dir)
    if utils.is_main_process() and degradation != 'none':
        print('[EVAL-DEG] degradation={}'.format(degradation))

    domain_records = []
    output_strs = []
    test_ap, test_acc = [], []

    for data_name, data_loader in data_loaders.items():
        local_rows = []
        sample_base = 0
        for batch_idx, samples in enumerate(data_loader):
            if getattr(args, 'eval_max_batches_per_domain', -1) > 0 and batch_idx >= args.eval_max_batches_per_domain:
                break
            images, labels = [sample.to(device, non_blocking=True) for sample in samples]
            outputs = model(images)
            if not isinstance(outputs, dict):
                # Defensive fallback; DREAM should return dict.
                probs = extract_logits(outputs).sigmoid().flatten()
                labels_cpu = labels.detach().cpu().view(-1).tolist()
                for i, (label, prob) in enumerate(zip(labels_cpu, probs.detach().cpu().tolist())):
                    local_rows.append({
                        'sample_index': sample_base + i,
                        'image_path': '',
                        'domain': data_name,
                        'degradation': degradation,
                        'label': int(label),
                        'p_final': float(prob),
                        'p_anchor': float(prob),
                        'z_final': float(np.log(max(prob, 1e-8) / max(1 - prob, 1e-8))),
                        'z_anchor': float(np.log(max(prob, 1e-8) / max(1 - prob, 1e-8))),
                        'pred_final': int(prob > 0.5),
                        'pred_anchor': int(prob > 0.5),
                        'final_correct': int((prob > 0.5) == bool(label)),
                        'anchor_correct': int((prob > 0.5) == bool(label)),
                        'ce_final': 0.0,
                        'ce_anchor': 0.0,
                        'help': 0,
                        'harm': 0,
                        'cavr': 0,
                        'decision_flip': 0,
                        'anchor_margin': abs(prob - 0.5) * 2,
                        'final_margin': abs(prob - 0.5) * 2,
                        'apply': 0.0,
                        'q_entropy': 0.0,
                        'correction': 0.0,
                        'best_candidate': 0,
                        'best_expert': 0,
                        'oracle_gain_ce': 0.0,
                        'final_oracle_gap_ce': 0.0,
                        'high_conf_anchor_harm': 0,
                    })
            else:
                local_rows.extend(build_prediction_rows(args, outputs, labels, data_name, degradation, sample_base))
            sample_base += labels.numel()

        rows = _gather_prediction_rows(local_rows)
        if utils.is_main_process():
            domain_record = domain_metrics_from_rows(args, rows, epoch, data_name, degradation)
            domain_records.append(domain_record)
            write_jsonl(args.output_dir, 'eval_domain_metrics.jsonl', domain_record)
            write_prediction_csv(args, rows, epoch, data_name, degradation)
            write_top_cases(args, rows, epoch, data_name, degradation)
            test_ap.append(domain_record.get('final_ap', np.nan))
            test_acc.append(domain_record.get('final_acc', np.nan))
            s = "({}) acc: {:.2f}; ap: {:.2f}; racc: {:.2f}; facc: {:.2f};".format(
                data_name,
                domain_record.get('final_acc', np.nan) * 100,
                domain_record.get('final_ap', np.nan) * 100,
                domain_record.get('final_racc', np.nan) * 100,
                domain_record.get('final_facc', np.nan) * 100,
            )
            output_strs.append(s)
            print(s)
            print("[DREAM-EVAL] domain={} deg={} final acc={:.2f} ap={:.2f} racc={:.2f} facc={:.2f}".format(
                data_name, degradation,
                domain_record.get('final_acc', np.nan) * 100,
                domain_record.get('final_ap', np.nan) * 100,
                domain_record.get('final_racc', np.nan) * 100,
                domain_record.get('final_facc', np.nan) * 100,
            ))
            print("[DREAM-EVAL] domain={} deg={} anchor acc={:.2f} ap={:.2f} racc={:.2f} facc={:.2f}".format(
                data_name, degradation,
                domain_record.get('anchor_acc', np.nan) * 100,
                domain_record.get('anchor_ap', np.nan) * 100,
                domain_record.get('anchor_racc', np.nan) * 100,
                domain_record.get('anchor_facc', np.nan) * 100,
            ))
            print("[DREAM-EVAL] domain={} deg={} cavr={:.4f} help={:.4f} harm={:.4f} apply={:.4f} q=[{:.4f},{:.4f},{:.4f}] corr={:.4f}".format(
                data_name, degradation,
                domain_record.get('cavr', np.nan),
                domain_record.get('help_rate', np.nan),
                domain_record.get('harm_rate', np.nan),
                domain_record.get('apply_mean', np.nan),
                domain_record.get('q_mean_e0', np.nan),
                domain_record.get('q_mean_e1', np.nan),
                domain_record.get('q_mean_e2', np.nan),
                domain_record.get('correction_abs_mean', np.nan),
            ))

    if utils.is_main_process():
        mean_record = mean_metrics_from_domains(args, domain_records, epoch, degradation)
        write_jsonl(args.output_dir, 'eval_mean_metrics.jsonl', mean_record)
        warning_flags(args, mean_record, epoch, degradation)
        output_strs.append("({:10}) acc: {:.2f}; ap: {:.2f}; racc: {:.2f}; facc: {:.2f};".format(
            'mean',
            mean_record.get('mean_acc_final', np.nan) * 100,
            mean_record.get('mean_ap_final', np.nan) * 100,
            mean_record.get('mean_racc_final', np.nan) * 100,
            mean_record.get('mean_facc_final', np.nan) * 100,
        ))
        print("[DREAM-EVAL-MEAN] deg={} final_acc={:.2f} final_ap={:.2f} anchor_acc={:.2f} anchor_ap={:.2f} cavr={:.4f} help={:.4f} harm={:.4f} apply={:.4f}".format(
            degradation,
            mean_record.get('mean_acc_final', np.nan) * 100,
            mean_record.get('mean_ap_final', np.nan) * 100,
            mean_record.get('mean_acc_anchor', np.nan) * 100,
            mean_record.get('mean_ap_anchor', np.nan) * 100,
            mean_record.get('mean_cavr', np.nan),
            mean_record.get('mean_help_rate', np.nan),
            mean_record.get('mean_harm_rate', np.nan),
            mean_record.get('mean_apply_mean', np.nan),
        ))
        cur_ap = mean_record.get('mean_ap_final', np.nan)
        cur_acc = mean_record.get('mean_acc_final', np.nan)
    else:
        mean_record, cur_ap, cur_acc = {}, 0.0, 0.0

    if degradation_name is not None:
        args.dream_eval_degradation = old_deg
    return '; '.join(output_strs), cur_ap, cur_acc, mean_record


@torch.no_grad()
def evaluate(model, data_loaders, device, args=None, test=False):
    if getattr(args, 'method', 'iapl') != 'dream_cs':
        output_strs, cur_ap, cur_acc, _ = _evaluate_iapl(model, data_loaders, device, args=args)
        return output_strs, cur_ap, cur_acc

    multi = getattr(args, 'dream_eval_multi_degradations', [])
    if len(multi) > 0:
        outputs = []
        per_deg = {}
        for name in multi:
            output_strs, cur_ap, cur_acc, mean_record = _evaluate_dream_once(
                model, data_loaders, device, args=args, degradation_name=name)
            outputs.append('[{}] {}'.format(name, output_strs))
            per_deg[name] = mean_record
        write_corruption_summary(args, per_deg, getattr(args, 'current_epoch', None))
        preferred = 'none' if 'none' in per_deg else multi[0]
        return ' | '.join(outputs), per_deg[preferred].get('mean_ap_final', 0.0), per_deg[preferred].get('mean_acc_final', 0.0)

    output_strs, cur_ap, cur_acc, _ = _evaluate_dream_once(model, data_loaders, device, args=args)
    return output_strs, cur_ap, cur_acc
