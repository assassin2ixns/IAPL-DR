import csv
import datetime
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import utils.misc as utils

try:
    from sklearn.metrics import average_precision_score, accuracy_score, roc_auc_score
except Exception:
    average_precision_score = None
    accuracy_score = None
    roc_auc_score = None


EPS = 1e-8
DEG_NAMES = ['jpeg90', 'jpeg75', 'jpeg50', 'resize', 'blur', 'quant', 'webp']


def is_rank0():
    return utils.is_main_process()


def ensure_output_dirs(output_dir):
    if not output_dir or not is_rank0():
        return
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'predictions').mkdir(parents=True, exist_ok=True)
    (root / 'cases').mkdir(parents=True, exist_ok=True)


def now_iso():
    return datetime.datetime.now().isoformat(timespec='seconds')


def json_safe(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return json_safe(value.detach().cpu().item())
        return [json_safe(v) for v in value.detach().cpu().flatten().tolist()]
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return value
        return float(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_jsonl(output_dir, filename, record):
    if not output_dir or not is_rank0():
        return
    ensure_output_dirs(output_dir)
    path = Path(output_dir) / filename
    with path.open('a') as f:
        f.write(json.dumps(json_safe(record), sort_keys=True) + '\n')


def write_json(output_dir, filename, record):
    if not output_dir or not is_rank0():
        return
    ensure_output_dirs(output_dir)
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        json.dump(json_safe(record), f, indent=2, sort_keys=True)


def base_record(args, epoch=None, iteration=None, global_step=None, split='train',
                domain='train', degradation='none', lr=None, batch_size=None):
    return {
        'timestamp': now_iso(),
        'epoch': epoch,
        'iter': iteration,
        'global_step': global_step,
        'split': split,
        'domain': domain,
        'degradation': degradation,
        'seed': getattr(args, 'seed', None),
        'model_name': getattr(args, 'model_name', None),
        'lr': lr,
        'batch_size': batch_size,
    }


def to_numpy(x):
    if x is None:
        return np.asarray([])
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def q_value(arr, q):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return float('nan')
    return float(np.quantile(arr, q))


def add_distribution(record, prefix, values, minmax=False):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        keys = ['mean', 'std', 'q10', 'q50', 'q90']
        if minmax:
            keys += ['min', 'max']
        for k in keys:
            record[f'{prefix}_{k}'] = float('nan')
        return
    record[f'{prefix}_mean'] = float(np.mean(arr))
    record[f'{prefix}_std'] = float(np.std(arr))
    record[f'{prefix}_q10'] = q_value(arr, 0.10)
    record[f'{prefix}_q50'] = q_value(arr, 0.50)
    record[f'{prefix}_q90'] = q_value(arr, 0.90)
    if minmax:
        record[f'{prefix}_min'] = float(np.min(arr))
        record[f'{prefix}_max'] = float(np.max(arr))


def safe_mean(arr):
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return float('nan')
    return float(np.mean(arr))


def corrcoef(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float('nan')
    a = a[mask]
    b = b[mask]
    if np.std(a) < EPS or np.std(b) < EPS:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def _ap(y_true, p):
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    if average_precision_score is None or y_true.size == 0:
        return float('nan')
    try:
        return float(average_precision_score(y_true, p))
    except Exception:
        return float('nan')


def _roc_auc(y_true, p):
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    if roc_auc_score is None or len(np.unique(y_true)) < 2:
        return float('nan')
    try:
        return float(roc_auc_score(y_true, p))
    except Exception:
        return float('nan')


def _brier(y_true, p):
    y_true = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if y_true.size == 0:
        return float('nan')
    return float(np.mean((p - y_true) ** 2))


def _nll(y_true, p):
    y_true = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    if y_true.size == 0:
        return float('nan')
    return float(np.mean(-(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))))


def _ece(y_true, p, n_bins=10):
    y_true = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if y_true.size == 0:
        return float('nan')
    conf = np.maximum(p, 1.0 - p)
    pred = (p > 0.5).astype(np.float64)
    correct = (pred == y_true).astype(np.float64)
    ece = 0.0
    for i in range(n_bins):
        low = i / n_bins
        high = (i + 1) / n_bins
        mask = (conf >= low) & (conf <= high if i == n_bins - 1 else conf < high)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return float(ece)


def _mce(y_true, p, n_bins=10):
    y_true = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if y_true.size == 0:
        return float('nan')
    conf = np.maximum(p, 1.0 - p)
    pred = (p > 0.5).astype(np.float64)
    correct = (pred == y_true).astype(np.float64)
    mce = 0.0
    for i in range(n_bins):
        low = i / n_bins
        high = (i + 1) / n_bins
        mask = (conf >= low) & (conf <= high if i == n_bins - 1 else conf < high)
        if mask.any():
            mce = max(mce, abs(float(correct[mask].mean()) - float(conf[mask].mean())))
    return float(mce)


def _best_threshold(y_true, p):
    y_true = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if y_true.size == 0:
        return float('nan'), float('nan'), float('nan')
    thresholds = np.unique(np.concatenate([np.linspace(0.0, 1.0, 101), p]))
    best_t, best_acc, best_bal = 0.5, -1.0, -1.0
    for t in thresholds:
        pred = (p > t).astype(np.float64)
        acc = float((pred == y_true).mean())
        real = y_true == 0
        fake = y_true == 1
        racc = float((pred[real] == y_true[real]).mean()) if real.any() else float('nan')
        facc = float((pred[fake] == y_true[fake]).mean()) if fake.any() else float('nan')
        bal = np.nanmean([racc, facc])
        if acc > best_acc:
            best_t, best_acc = float(t), acc
        if bal > best_bal:
            best_bal = float(bal)
    return best_t, best_acc, best_bal


def binary_metrics(y_true, p, prefix=''):
    y_true = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    out = {}
    if y_true.size == 0:
        return out
    pred = (p > 0.5).astype(np.float64)
    real = y_true == 0
    fake = y_true == 1
    best_t, best_acc, best_bal = _best_threshold(y_true, p)
    key = lambda name: f'{prefix}_{name}' if prefix else name
    out[key('acc')] = float((pred == y_true).mean())
    out[key('ap')] = _ap(y_true, p)
    out[key('roc_auc')] = _roc_auc(y_true, p)
    out[key('racc')] = float((pred[real] == y_true[real]).mean()) if real.any() else float('nan')
    out[key('facc')] = float((pred[fake] == y_true[fake]).mean()) if fake.any() else float('nan')
    out[key('rf_gap')] = abs(out[key('racc')] - out[key('facc')])
    out[key('brier')] = _brier(y_true, p)
    out[key('ece_10')] = _ece(y_true, p, 10)
    out[key('ece_15')] = _ece(y_true, p, 15)
    out[key('mce_10')] = _mce(y_true, p, 10)
    out[key('nll')] = _nll(y_true, p)
    out[key('pos_rate')] = float(pred.mean())
    out[key('prob_mean')] = float(p.mean())
    out[key('prob_std')] = float(p.std())
    out[key('prob_real_mean')] = float(p[real].mean()) if real.any() else float('nan')
    out[key('prob_fake_mean')] = float(p[fake].mean()) if fake.any() else float('nan')
    logits = np.log(np.clip(p, EPS, 1.0 - EPS) / np.clip(1.0 - p, EPS, 1.0))
    out[key('logit_mean')] = float(logits.mean())
    out[key('logit_std')] = float(logits.std())
    out[key('best_threshold_diag')] = best_t
    out[key('best_acc_diag')] = best_acc
    out[key('best_balanced_acc_diag')] = best_bal
    return out


def _ce_from_prob(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def _safe_torch_mean(x):
    if x.numel() == 0:
        return float('nan')
    return float(x.detach().float().mean().cpu().item())


def _batch_acc(probs, labels):
    if probs.numel() == 0:
        return float('nan')
    return float(((probs > 0.5).float() == labels.float()).float().mean().detach().cpu().item())


def _param_norm_named(model, include_substr):
    total = None
    for name, param in model.named_parameters():
        if include_substr in name:
            value = param.detach().float().norm(2).pow(2)
            total = value if total is None else total + value
    if total is None:
        return 0.0
    return float(total.sqrt().cpu().item())


def _grad_norm_named(model, include_substr):
    total = None
    for name, param in model.named_parameters():
        if include_substr in name and param.grad is not None:
            value = param.grad.detach().float().norm(2).pow(2)
            total = value if total is None else total + value
    if total is None:
        return 0.0
    return float(total.sqrt().cpu().item())


def _batch_threshold_metrics(record, prefix, logits, labels):
    labels = labels.float().view(-1)
    probs = logits.detach().float().view(-1).sigmoid()
    real = labels == 0
    fake = labels == 1
    acc = _batch_acc(probs, labels)
    racc = _batch_acc(probs[real], labels[real])
    facc = _batch_acc(probs[fake], labels[fake])
    record[f'train_acc_{prefix}'] = acc
    record[f'train_racc_{prefix}'] = racc
    record[f'train_facc_{prefix}'] = facc
    record[f'train_rf_gap_{prefix}'] = abs(racc - facc) if math.isfinite(racc) and math.isfinite(facc) else float('nan')
    record[f'{prefix}_real_prob_mean'] = _safe_torch_mean(probs[real])
    record[f'{prefix}_fake_prob_mean'] = _safe_torch_mean(probs[fake])
    record[f'{prefix}_real_logit_mean'] = _safe_torch_mean(logits.detach().float().view(-1)[real])
    record[f'{prefix}_fake_logit_mean'] = _safe_torch_mean(logits.detach().float().view(-1)[fake])


def compute_train_record(args, outputs, labels, loss_dict, weight_dict, model, optimizer,
                         epoch, iteration, global_step, data_time, batch_time):
    lr = optimizer.param_groups[0]['lr']
    labels_f = labels.detach().float().view(-1)
    z = outputs['logits_flat'].detach().float().view(-1)
    z0 = outputs['anchor_logits'].detach().float().view(-1)
    ze = outputs['expert_logits'].detach().float()
    q = outputs['q'].detach().float()
    apply_eff = outputs.get('apply', outputs.get('apply_eff')).detach().float().view(-1)
    apply_raw = outputs.get('apply_raw', apply_eff).detach().float().view(-1)
    delta = outputs.get('delta_raw', ze - z0[:, None]).detach().float()
    delta_clamped = outputs.get('delta_clamped', delta).detach().float()
    p = z.sigmoid()
    p0 = z0.sigmoid()
    pe = ze.sigmoid()
    k = ze.shape[1]
    record = base_record(args, epoch, iteration, global_step, 'train', 'train', 'none',
                         lr=lr, batch_size=int(labels_f.numel()))

    loss_total = 0.0
    nan_flag = False
    for name, value in loss_dict.items():
        if name.startswith('stat_'):
            continue
        scalar = float(value.detach().float().cpu().item())
        record[name] = scalar
        if name in weight_dict:
            loss_total += scalar * float(weight_dict[name])
        if not math.isfinite(scalar):
            nan_flag = True
    record['loss_total'] = loss_total
    record['lr'] = lr
    record['data_time'] = data_time
    record['batch_time'] = batch_time
    record['max_gpu_mem_mb'] = float(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0) if torch.cuda.is_available() else 0.0
    record['nan_or_inf_flag'] = bool(nan_flag or (not math.isfinite(loss_total)))
    record['dream_fast_mode'] = outputs.get('fast_mode', getattr(args, 'dream_fast_mode', 'off'))
    record['dream_fast_readout'] = outputs.get('fast_readout', getattr(args, 'dream_fast_readout', 'batchfold'))
    record['dream_anchor_purity'] = outputs.get('anchor_purity', 'pure_anchor')
    for src_key, dst_key in [
        ('encoder_calls', 'dream_encoder_calls'),
        ('encoder_calls_expected', 'dream_encoder_calls_expected'),
        ('prompt_bank_len', 'dream_prompt_bank_len'),
        ('effective_encoder_multiplier', 'dream_effective_encoder_multiplier'),
    ]:
        value = outputs.get(src_key, None)
        if torch.is_tensor(value):
            record[dst_key] = float(value.detach().float().cpu().item())
        elif value is not None:
            record[dst_key] = float(value)
    scale = outputs.get('fast_prompt_delta_scale', None)
    if scale is not None:
        scale = to_numpy(scale).reshape(-1)
        record['dream_fast_prompt_delta_scale_mean'] = safe_mean(scale)
        for idx in range(min(3, scale.shape[0])):
            record[f'dream_fast_prompt_delta_scale_e{idx}'] = float(scale[idx])
    prompt_delta_norm = outputs.get('prompt_delta_norm_per_expert', None)
    if prompt_delta_norm is not None:
        prompt_delta_norm = to_numpy(prompt_delta_norm).reshape(-1)
        for idx in range(min(3, prompt_delta_norm.shape[0])):
            record[f'dream_prompt_delta_norm_e{idx}'] = float(prompt_delta_norm[idx])
    prompt_delta_logit = outputs.get('prompt_delta_logit_per_expert', None)
    if prompt_delta_logit is not None:
        prompt_delta_logit = to_numpy(prompt_delta_logit).reshape(-1)
        for idx in range(min(3, prompt_delta_logit.shape[0])):
            record[f'dream_prompt_delta_logit_e{idx}'] = float(prompt_delta_logit[idx])
    bank_cls = outputs.get('bank_cls', None)
    if bank_cls is not None:
        bank_cls_f = bank_cls.detach().float()
        record['bank_cls_norm'] = _safe_torch_mean(bank_cls_f.pow(2).mean(dim=-1).sqrt())
    p0_bank = outputs.get('prompt_bank_anchor_feature', None)
    if p0_bank is not None:
        p0_f = p0_bank.detach().float()
        record['p0_norm'] = _safe_torch_mean(p0_f.pow(2).mean(dim=-1).sqrt())
    pe_bank = outputs.get('prompt_bank_expert_feature', None)
    if pe_bank is not None:
        pe_f = pe_bank.detach().float()
        pe_norm = pe_f.pow(2).mean(dim=-1).sqrt()
        for idx in range(min(3, pe_norm.shape[1])):
            record[f'pe_norm_e{idx}'] = _safe_torch_mean(pe_norm[:, idx])
    clean_loss = abs(record.get('loss_dream_clean', 0.0)) + EPS
    record['ratio_rob_to_clean'] = record.get('loss_dream_rob', 0.0) / clean_loss
    record['ratio_clean_safe_to_clean'] = record.get('loss_dream_clean_safe', 0.0) / clean_loss
    record['ratio_expert_to_clean'] = record.get('loss_dream_expert', 0.0) / clean_loss
    record['ratio_route_to_clean'] = record.get('loss_dream_route', 0.0) / clean_loss

    add_distribution(record, 'prob_final', to_numpy(p), True)
    add_distribution(record, 'prob_anchor', to_numpy(p0), True)
    add_distribution(record, 'logit_final', to_numpy(z))
    add_distribution(record, 'logit_anchor', to_numpy(z0))
    record['pos_rate_final'] = _safe_torch_mean((p > 0.5).float())
    record['pos_rate_anchor'] = _safe_torch_mean((p0 > 0.5).float())
    record['label_pos_rate'] = _safe_torch_mean(labels_f)
    record['final_anchor_prob_gap'] = _safe_torch_mean((p - p0).abs())
    record['final_anchor_logit_gap'] = _safe_torch_mean((z - z0).abs())
    _batch_threshold_metrics(record, 'final', z, labels_f)
    _batch_threshold_metrics(record, 'anchor', z0, labels_f)

    for idx in range(min(3, k)):
        add_distribution(record, f'prob_expert{idx}', to_numpy(pe[:, idx]))
        add_distribution(record, f'logit_expert{idx}', to_numpy(ze[:, idx]))
        record[f'pos_rate_expert{idx}'] = _safe_torch_mean((pe[:, idx] > 0.5).float())
        record[f'expert_anchor_prob_gap{idx}'] = _safe_torch_mean((pe[:, idx] - p0).abs())
        record[f'expert_anchor_logit_gap{idx}'] = _safe_torch_mean((ze[:, idx] - z0).abs())
        _batch_threshold_metrics(record, f'expert{idx}', ze[:, idx], labels_f)

    ce_final = F.binary_cross_entropy_with_logits(z, labels_f, reduction='none')
    ce_anchor = F.binary_cross_entropy_with_logits(z0, labels_f, reduction='none')
    ce_expert = F.binary_cross_entropy_with_logits(ze, labels_f[:, None].expand_as(ze), reduction='none')
    pred_final = p > 0.5
    pred_anchor = p0 > 0.5
    correct_final = pred_final == labels_f.bool()
    correct_anchor = pred_anchor == labels_f.bool()
    anchor_margin = (p0 - 0.5).abs() * 2.0
    final_margin = (p - 0.5).abs() * 2.0
    flip = pred_final != pred_anchor
    record['ce_final_mean'] = _safe_torch_mean(ce_final)
    record['ce_anchor_mean'] = _safe_torch_mean(ce_anchor)
    record['ce_delta_final_minus_anchor'] = _safe_torch_mean(ce_final - ce_anchor)
    cavr_eps = float(getattr(args, 'dream_cavr_eps', 1e-4))
    record['cavr_raw'] = _safe_torch_mean((ce_final > ce_anchor).float())
    record['cavr_eps'] = _safe_torch_mean((ce_final > ce_anchor + cavr_eps).float())
    record['cavr_strict'] = _safe_torch_mean((ce_final > ce_anchor + 1e-3).float())
    record['cavr'] = record['cavr_eps']
    margin = float(getattr(args, 'dream_clean_safe_margin', 0.0))
    record['clean_safe_violation_rate'] = _safe_torch_mean((ce_final > ce_anchor.detach() + margin).float())
    record['help_rate'] = _safe_torch_mean((correct_final & ~correct_anchor).float())
    record['harm_rate'] = _safe_torch_mean((~correct_final & correct_anchor).float())
    record['both_correct_rate'] = _safe_torch_mean((correct_final & correct_anchor).float())
    record['both_wrong_rate'] = _safe_torch_mean((~correct_final & ~correct_anchor).float())
    record['no_regret_index'] = record['help_rate'] - record['harm_rate']
    record['help_harm_ratio'] = record['help_rate'] / (record['harm_rate'] + EPS)
    record['final_minus_anchor_acc'] = record['train_acc_final'] - record['train_acc_anchor']
    record['final_minus_anchor_racc'] = record['train_racc_final'] - record['train_racc_anchor']
    record['final_minus_anchor_facc'] = record['train_facc_final'] - record['train_facc_anchor']
    record['decision_flip_rate'] = _safe_torch_mean(flip.float())
    record['flip_help_rate'] = _safe_torch_mean((flip & correct_final).float())
    record['flip_harm_rate'] = _safe_torch_mean((flip & ~correct_final & correct_anchor).float())
    record['high_conf_anchor_harm_rate'] = _safe_torch_mean((correct_anchor & (anchor_margin > 0.8) & ~correct_final).float())
    record['high_conf_anchor_flip_rate'] = _safe_torch_mean(((anchor_margin > 0.8) & flip).float())

    add_distribution(record, 'apply_raw', to_numpy(apply_raw), True)
    add_distribution(record, 'apply_eff', to_numpy(apply_eff), True)
    record['apply_rate_gt_001'] = _safe_torch_mean((apply_eff > 0.01).float())
    record['apply_rate_gt_005'] = _safe_torch_mean((apply_eff > 0.05).float())
    record['apply_rate_gt_010'] = _safe_torch_mean((apply_eff > 0.10).float())
    record['apply_rate_gt_050'] = _safe_torch_mean((apply_eff > 0.50).float())
    real = labels_f == 0
    fake = labels_f == 1
    record['apply_real_mean'] = _safe_torch_mean(apply_eff[real])
    record['apply_fake_mean'] = _safe_torch_mean(apply_eff[fake])
    record['apply_correct_anchor_mean'] = _safe_torch_mean(apply_eff[correct_anchor])
    record['apply_wrong_anchor_mean'] = _safe_torch_mean(apply_eff[~correct_anchor])

    q_entropy = -(q.clamp_min(EPS) * q.clamp_min(EPS).log()).sum(dim=1)
    q_top1 = q.argmax(dim=1)
    record['q_entropy_mean'] = _safe_torch_mean(q_entropy)
    record['q_entropy_norm'] = record['q_entropy_mean'] / math.log(max(k, 2))
    record['q_max_mean'] = _safe_torch_mean(q.max(dim=1).values)
    record['q_min_mean'] = _safe_torch_mean(q.min(dim=1).values)
    for idx in range(min(3, k)):
        record[f'q_mean_e{idx}'] = _safe_torch_mean(q[:, idx])
        record[f'q_std_e{idx}'] = float(q[:, idx].detach().float().std(unbiased=False).cpu().item())
        record[f'q_top1_rate_e{idx}'] = _safe_torch_mean((q_top1 == idx).float())
        record[f'q_real_mean_e{idx}'] = _safe_torch_mean(q[real, idx])
        record[f'q_fake_mean_e{idx}'] = _safe_torch_mean(q[fake, idx])
        record[f'q_anchor_correct_mean_e{idx}'] = _safe_torch_mean(q[correct_anchor, idx])
        record[f'q_anchor_wrong_mean_e{idx}'] = _safe_torch_mean(q[~correct_anchor, idx])

    i_clean = ce_anchor.detach()[:, None] - ce_expert.detach()
    anchor_worst = ce_anchor.detach()
    expert_worst = ce_expert.detach()
    if outputs.get('deg_outputs'):
        anchor_views = [ce_anchor.detach()]
        expert_views = [ce_expert.detach()]
        for d in outputs['deg_outputs']:
            dz0 = d['anchor_logits'].detach().float().view(-1)
            dze = d['expert_logits'].detach().float()
            anchor_views.append(F.binary_cross_entropy_with_logits(dz0, labels_f, reduction='none').detach())
            expert_views.append(F.binary_cross_entropy_with_logits(dze, labels_f[:, None].expand_as(dze), reduction='none').detach())
        anchor_worst = torch.stack(anchor_views, 0).max(0).values
        expert_worst = torch.stack(expert_views, 0).max(0).values
    i_rob = anchor_worst[:, None] - expert_worst
    improvement = torch.minimum(i_clean, i_rob)
    active = improvement.max(dim=1).values > float(getattr(args, 'dream_route_margin', 0.0))
    raw_target = torch.relu(improvement / max(float(getattr(args, 'dream_route_tau', 0.5)), EPS))
    target_q = torch.softmax(raw_target, dim=1)
    target_entropy = -(target_q.clamp_min(EPS) * target_q.clamp_min(EPS).log()).sum(dim=1)
    record['active_rate'] = _safe_torch_mean(active.float())
    record['target_q_entropy'] = _safe_torch_mean(target_entropy)
    record['route_top1_agreement'] = _safe_torch_mean((q.argmax(1)[active] == target_q.argmax(1)[active]).float()) if active.any() else float('nan')
    record['route_kl_active'] = _safe_torch_mean(F.kl_div((q[active].clamp_min(EPS)).log(), target_q[active], reduction='none').sum(1)) if active.any() else float('nan')
    record['apply_target_mean'] = _safe_torch_mean(active.float())
    mask_apply = apply_eff > 0.05
    record['apply_target_precision_at_005'] = _safe_torch_mean(active[mask_apply].float()) if mask_apply.any() else float('nan')
    record['apply_target_recall_at_005'] = _safe_torch_mean(mask_apply[active].float()) if active.any() else float('nan')
    record['corr_apply_anchor_uncertainty'] = corrcoef(to_numpy(apply_eff), to_numpy(1.0 - anchor_margin))
    oracle_gain = ce_anchor.detach() - torch.min(ce_expert.detach(), dim=1).values
    record['corr_apply_oracle_gain'] = corrcoef(to_numpy(apply_eff), to_numpy(oracle_gain))
    for idx in range(min(3, k)):
        record[f'target_q_mean_e{idx}'] = _safe_torch_mean(target_q[:, idx])
        record[f'i_clean_e{idx}'] = _safe_torch_mean(i_clean[:, idx])
        record[f'i_rob_e{idx}'] = _safe_torch_mean(i_rob[:, idx])
        record[f'corr_qe_ie_clean_e{idx}'] = corrcoef(to_numpy(q[:, idx]), to_numpy(i_clean[:, idx]))
        record[f'corr_qe_ie_rob_e{idx}'] = corrcoef(to_numpy(q[:, idx]), to_numpy(i_rob[:, idx]))

    ce_candidates = torch.cat([ce_anchor[:, None], ce_expert], dim=1)
    best_candidate = ce_candidates.argmin(dim=1)
    best_expert = ce_expert.argmin(dim=1)
    oracle_ce = ce_candidates.min(dim=1).values
    expert_oracle_ce = ce_expert.min(dim=1).values
    cand_logits = torch.cat([z0[:, None], ze], dim=1)
    cand_pred = cand_logits.gather(1, best_candidate[:, None]).squeeze(1).sigmoid() > 0.5
    exp_pred = ze.gather(1, best_expert[:, None]).squeeze(1).sigmoid() > 0.5
    record['candidate_oracle_acc'] = _safe_torch_mean((cand_pred == labels_f.bool()).float())
    record['expert_oracle_acc'] = _safe_torch_mean((exp_pred == labels_f.bool()).float())
    record['expert_oracle_gain_ce'] = _safe_torch_mean(ce_anchor - expert_oracle_ce)
    record['candidate_oracle_gain_ce'] = _safe_torch_mean(ce_anchor - oracle_ce)
    record['final_oracle_gap_ce'] = _safe_torch_mean(ce_final - oracle_ce)
    good_oracle = (ce_anchor - oracle_ce) > 0.05
    capture = ((ce_anchor - ce_final) / (ce_anchor - oracle_ce + EPS)).clamp(-5, 5)
    record['router_capture_rate'] = _safe_torch_mean(capture[good_oracle]) if good_oracle.any() else float('nan')
    record['best_candidate_rate_anchor'] = _safe_torch_mean((best_candidate == 0).float())
    for idx in range(min(3, k)):
        record[f'ce_expert{idx}_mean'] = _safe_torch_mean(ce_expert[:, idx])
        record[f'expert{idx}_minus_anchor_ce'] = _safe_torch_mean(ce_expert[:, idx] - ce_anchor)
        record[f'expert{idx}_better_than_anchor_rate'] = _safe_torch_mean((ce_expert[:, idx] < ce_anchor).float())
        record[f'best_candidate_rate_expert{idx}'] = _safe_torch_mean((best_candidate == idx + 1).float())
        record[f'best_expert_rate_e{idx}'] = _safe_torch_mean((best_expert == idx).float())

    correction = z - z0
    add_distribution(record, 'correction_abs', to_numpy(correction.abs()))
    record['correction_signed_mean'] = _safe_torch_mean(correction)
    record['correction_real_mean'] = _safe_torch_mean(correction[real])
    record['correction_fake_mean'] = _safe_torch_mean(correction[fake])
    record['correction_correct_anchor_mean'] = _safe_torch_mean(correction[correct_anchor])
    record['correction_wrong_anchor_mean'] = _safe_torch_mean(correction[~correct_anchor])
    record['correction_flip_mean'] = _safe_torch_mean(correction[flip].abs()) if flip.any() else float('nan')
    for idx in range(min(3, k)):
        record[f'delta_abs_e{idx}'] = _safe_torch_mean(delta[:, idx].abs())
        record[f'delta_signed_e{idx}'] = _safe_torch_mean(delta[:, idx])
        record[f'delta_clamped_abs_e{idx}'] = _safe_torch_mean(delta_clamped[:, idx].abs())
        record[f'delta_clip_frac_e{idx}'] = _safe_torch_mean((delta[:, idx].abs() >= float(getattr(args, 'dream_delta_clip', 1.0)) - 1e-6).float())
        record[f'delta_real_e{idx}'] = _safe_torch_mean(delta[real, idx])
        record[f'delta_fake_e{idx}'] = _safe_torch_mean(delta[fake, idx])

    record['residual_norm_mean'] = float(outputs['prompt_residual_norm'].detach().cpu().item())
    record['residual_cos_offdiag_mean'] = float(outputs.get('residual_cos_offdiag', torch.tensor(float('nan'))).detach().cpu().item())
    record['residual_diversity_loss'] = record.get('loss_dream_div', float('nan'))
    residual_norms = outputs.get('residual_norm_per_expert')
    residual_scales = outputs.get('residual_scale_per_expert')
    feature_norms = outputs.get('feature_residual_norm_per_expert')
    if residual_norms is not None:
        rn = residual_norms.detach().float().cpu().view(-1).tolist()
        for idx in range(min(3, len(rn))):
            record[f'residual_norm_e{idx}'] = rn[idx]
    if residual_scales is not None:
        rs = residual_scales.detach().float().cpu().view(-1).tolist()
        for idx in range(min(3, len(rs))):
            record[f'residual_scale_e{idx}'] = rs[idx]
            record[f'residual_scale_abs_e{idx}'] = abs(rs[idx])
    if hasattr(model, 'dream_expert_bank'):
        with torch.no_grad():
            rmat = model.dream_expert_bank.residual_matrix().detach().float().view(k, -1)
            rnorm = F.normalize(rmat, dim=1, eps=EPS)
            cos = torch.matmul(rnorm, rnorm.t()).cpu().numpy()
        pairs = [(0, 1), (0, 2), (1, 2)]
        for a, b in pairs:
            if a < k and b < k:
                record[f'residual_cos_e{a}e{b}'] = float(cos[a, b])
    if feature_norms is not None:
        fn = feature_norms.detach().float().cpu()
        for idx in range(min(3, fn.shape[1])):
            record[f'feature_residual_norm_e{idx}'] = float(fn[:, idx].mean().item())
    he = outputs.get('he')
    if he is not None:
        h = he.detach().float()
        for a, b in [(0, 1), (0, 2), (1, 2)]:
            if a < k and b < k:
                c = F.cosine_similarity(h[:, a], h[:, b], dim=1)
                record[f'feature_residual_cos_e{a}e{b}'] = _safe_torch_mean(c)

    grad_router = _grad_norm_named(model, 'dream_router')
    grad_a = _grad_norm_named(model, 'dream_expert_bank.A')
    grad_b = _grad_norm_named(model, 'dream_expert_bank.B')
    grad_scale = _grad_norm_named(model, 'dream_expert_bank.scale')
    grad_prompt = _grad_norm_named(model, 'prompt_learner.ctx')
    record['grad_norm_dream_router'] = grad_router
    record['grad_norm_dream_expert_bank_A'] = grad_a
    record['grad_norm_dream_expert_bank_B'] = grad_b
    record['grad_norm_dream_expert_scale'] = grad_scale
    record['grad_norm_prompt_learner_ctx'] = grad_prompt
    record['grad_norm_conditional_ctx'] = _grad_norm_named(model, 'conditional_ctx')
    record['grad_norm_fc_binary'] = _grad_norm_named(model, 'fc_binary')
    record['grad_norm_adapters'] = _grad_norm_named(model, 'adapter')
    record['update_ratio_expert_scale'] = lr * grad_scale / (_param_norm_named(model, 'dream_expert_bank.scale') + EPS)
    record['update_ratio_router'] = lr * grad_router / (_param_norm_named(model, 'dream_router') + EPS)
    record['update_ratio_prompt_ctx'] = lr * grad_prompt / (_param_norm_named(model, 'prompt_learner.ctx') + EPS)

    deg_outputs = outputs.get('deg_outputs', [])
    record['deg_num_views'] = len(deg_outputs)
    for name in DEG_NAMES:
        record[f'deg_count_{name}'] = 0
    if deg_outputs:
        final_views = [z]
        anchor_views = [z0]
        final_ce_views = [ce_final]
        anchor_ce_views = [ce_anchor]
        view_names = ['clean']
        for d in deg_outputs:
            name = d.get('deg_name', 'unknown')
            if name in DEG_NAMES:
                record[f'deg_count_{name}'] += 1
            dz = d['logits_flat'].detach().float().view(-1)
            dz0 = d['anchor_logits'].detach().float().view(-1)
            final_views.append(dz)
            anchor_views.append(dz0)
            final_ce_views.append(F.binary_cross_entropy_with_logits(dz, labels_f, reduction='none'))
            anchor_ce_views.append(F.binary_cross_entropy_with_logits(dz0, labels_f, reduction='none'))
            view_names.append(name)
            record[f'deg_{name}_ce_final'] = _safe_torch_mean(final_ce_views[-1])
            record[f'deg_{name}_ce_anchor'] = _safe_torch_mean(anchor_ce_views[-1])
            record[f'deg_{name}_acc_final'] = _batch_acc(dz.sigmoid(), labels_f)
            record[f'deg_{name}_acc_anchor'] = _batch_acc(dz0.sigmoid(), labels_f)
            record[f'deg_{name}_apply_mean'] = _safe_torch_mean(d.get('apply', apply_eff).detach().float().view(-1))
            record[f'deg_{name}_correction_abs'] = _safe_torch_mean((dz - dz0).abs())
            record[f'deg_{name}_logit_diff_final_clean'] = _safe_torch_mean((dz - z).abs())
            record[f'deg_{name}_logit_diff_anchor_clean'] = _safe_torch_mean((dz0 - z0).abs())
            dq = d.get('q')
            if dq is not None:
                dq = dq.detach().float()
                for idx in range(min(3, dq.shape[1])):
                    record[f'deg_{name}_q_mean_e{idx}'] = _safe_torch_mean(dq[:, idx])
            dze = d['expert_logits'].detach().float()
            dce = F.binary_cross_entropy_with_logits(dze, labels_f[:, None].expand_as(dze), reduction='none')
            for idx in range(min(3, dce.shape[1])):
                record[f'avg_deg_ce_expert{idx}'] = record.get(f'avg_deg_ce_expert{idx}', 0.0) + _safe_torch_mean(dce[:, idx]) / len(deg_outputs)
        f_stack = torch.stack(final_views, 0)
        a_stack = torch.stack(anchor_views, 0)
        p_stack = f_stack.sigmoid()
        pa_stack = a_stack.sigmoid()
        record['clean_deg_logit_var_final'] = _safe_torch_mean(f_stack.var(dim=0, unbiased=False))
        record['clean_deg_logit_var_anchor'] = _safe_torch_mean(a_stack.var(dim=0, unbiased=False))
        record['clean_deg_prob_var_final'] = _safe_torch_mean(p_stack.var(dim=0, unbiased=False))
        record['clean_deg_prob_var_anchor'] = _safe_torch_mean(pa_stack.var(dim=0, unbiased=False))
        final_ce_stack = torch.stack(final_ce_views, 0)
        anchor_ce_stack = torch.stack(anchor_ce_views, 0)
        record['worst_view_ce_final'] = _safe_torch_mean(final_ce_stack.max(0).values)
        record['worst_view_ce_anchor'] = _safe_torch_mean(anchor_ce_stack.max(0).values)
        record['avg_deg_ce_final'] = _safe_torch_mean(torch.stack(final_ce_views[1:], 0).mean(0))
        record['avg_deg_ce_anchor'] = _safe_torch_mean(torch.stack(anchor_ce_views[1:], 0).mean(0))
        worst_final_idx = final_ce_stack.mean(1).argmax().item()
        worst_anchor_idx = anchor_ce_stack.mean(1).argmax().item()
        record[f'worst_view_name_final_count_{view_names[worst_final_idx]}'] = 1
        record[f'worst_view_name_anchor_count_{view_names[worst_anchor_idx]}'] = 1
        light_logits = [ze[:, 0]]
        strong_logits = []
        consensus_logits = [ze[:, min(2, k - 1)]]
        for d in deg_outputs:
            name = d.get('deg_name', '')
            dze = d['expert_logits'].detach().float()
            if name in ['jpeg90', 'jpeg75'] and k > 0:
                light_logits.append(dze[:, 0])
            if name in ['jpeg50', 'resize', 'blur', 'quant', 'webp'] and k > 1:
                strong_logits.append(dze[:, 1])
            if k > 2:
                consensus_logits.append(dze[:, 2])
        record['fine_light_ce'] = _safe_torch_mean(F.binary_cross_entropy_with_logits(torch.stack(light_logits, 0), labels_f[None, :].expand(len(light_logits), -1), reduction='none').mean(0)) if light_logits else float('nan')
        record['stable_strong_ce'] = _safe_torch_mean(torch.stack([F.binary_cross_entropy_with_logits(s, labels_f, reduction='none') for s in strong_logits], 0).max(0).values) if strong_logits else float('nan')
        record['consensus_view_var'] = _safe_torch_mean(torch.stack(consensus_logits, 0).var(0, unbiased=False)) if consensus_logits else float('nan')
        record['specialize_fine_loss'] = record['fine_light_ce']
        record['specialize_stable_loss'] = record['stable_strong_ce']
        record['specialize_consensus_loss'] = record['consensus_view_var']

    return record


def concise_console_record(record):
    keys = [
        'dream_fast_mode', 'dream_encoder_calls',
        'loss_total', 'lr', 'train_acc_final', 'train_acc_anchor',
        'train_racc_final', 'train_facc_final', 'apply_eff_mean',
        'q_entropy_mean', 'correction_abs_mean', 'cavr', 'help_rate',
        'harm_rate', 'expert_oracle_gain_ce', 'final_oracle_gap_ce',
        'active_rate', 'residual_cos_offdiag_mean',
        'max_gpu_mem_mb',
    ]
    return {k: record[k] for k in keys if k in record}


def aggregate_records(records, args, epoch):
    out = base_record(args, epoch, None, None, 'train_epoch', 'train', 'none')
    if not records:
        return out
    keys = sorted({k for r in records for k, v in r.items() if isinstance(v, (int, float, bool, np.floating, np.integer))})
    for key in keys:
        vals = [float(r[key]) for r in records if key in r and isinstance(r[key], (int, float, bool, np.floating, np.integer))]
        if vals:
            out[key] = float(np.nanmean(vals))
    out['num_logged_batches'] = len(records)
    return out


def debug_first_batch(args, outputs, labels, epoch):
    if getattr(args, 'method', 'iapl') != 'dream_cs':
        return
    labels_f = labels.detach().float().view(-1)
    record = base_record(args, epoch, 0, epoch, 'train_debug', 'train', 'none',
                         batch_size=int(labels_f.numel()))
    record['label_values'] = labels.detach().cpu().view(-1).tolist()[:64]
    tensors = {
        'logits_final': outputs['logits_flat'],
        'logits_anchor': outputs['anchor_logits'],
        'logits_expert': outputs['expert_logits'],
        'apply': outputs.get('apply'),
        'apply_raw': outputs.get('apply_raw'),
        'q': outputs.get('q'),
        'delta_raw': outputs.get('delta_raw'),
        'delta_clamped': outputs.get('delta_clamped'),
    }
    for name, tensor in tensors.items():
        if tensor is not None:
            arr = to_numpy(tensor).reshape(-1)
            add_distribution(record, name, arr, True)
            record[f'{name}_shape'] = list(tensor.shape)
    write_json(getattr(args, 'output_dir', None), f'debug_first_batch_epoch_{epoch}.json', record)


def build_prediction_rows(args, outputs, labels, domain, degradation, start_index=0):
    labels_np = to_numpy(labels).reshape(-1).astype(int)
    z = to_numpy(outputs['logits_flat']).reshape(-1)
    z0 = to_numpy(outputs['anchor_logits']).reshape(-1)
    ze = to_numpy(outputs['expert_logits'])
    q = to_numpy(outputs['q'])
    apply_eff = to_numpy(outputs.get('apply')).reshape(-1)
    delta = to_numpy(outputs.get('delta_raw', outputs['expert_logits'] - outputs['anchor_logits'][:, None]))
    delta_c = to_numpy(outputs.get('delta_clamped', outputs['expert_logits'] - outputs['anchor_logits'][:, None]))
    p = 1.0 / (1.0 + np.exp(-z))
    p0 = 1.0 / (1.0 + np.exp(-z0))
    pe = 1.0 / (1.0 + np.exp(-ze))
    ce_f = _ce_from_prob(labels_np, p)
    ce_a = _ce_from_prob(labels_np, p0)
    ce_e = np.stack([_ce_from_prob(labels_np, pe[:, i]) for i in range(pe.shape[1])], axis=1)
    best_candidate = np.argmin(np.concatenate([ce_a[:, None], ce_e], axis=1), axis=1)
    best_expert = np.argmin(ce_e, axis=1)
    oracle = np.min(np.concatenate([ce_a[:, None], ce_e], axis=1), axis=1)
    pred_f = (p > 0.5).astype(int)
    pred_a = (p0 > 0.5).astype(int)
    pred_e = (pe > 0.5).astype(int)
    rows = []
    cavr_eps = float(getattr(args, 'dream_cavr_eps', 1e-4))
    for i in range(len(labels_np)):
        row = {
            'sample_index': int(start_index + i),
            'image_path': '',
            'domain': domain,
            'degradation': degradation,
            'label': int(labels_np[i]),
            'p_final': float(p[i]),
            'p_anchor': float(p0[i]),
            'z_final': float(z[i]),
            'z_anchor': float(z0[i]),
            'pred_final': int(pred_f[i]),
            'pred_anchor': int(pred_a[i]),
            'final_correct': int(pred_f[i] == labels_np[i]),
            'anchor_correct': int(pred_a[i] == labels_np[i]),
            'ce_final': float(ce_f[i]),
            'ce_anchor': float(ce_a[i]),
            'help': int(pred_f[i] == labels_np[i] and pred_a[i] != labels_np[i]),
            'harm': int(pred_f[i] != labels_np[i] and pred_a[i] == labels_np[i]),
            'cavr_raw': int(ce_f[i] > ce_a[i]),
            'cavr_eps': int(ce_f[i] > ce_a[i] + cavr_eps),
            'cavr_strict': int(ce_f[i] > ce_a[i] + 1e-3),
            'cavr': int(ce_f[i] > ce_a[i] + cavr_eps),
            'decision_flip': int(pred_f[i] != pred_a[i]),
            'anchor_margin': float(abs(p0[i] - 0.5) * 2.0),
            'final_margin': float(abs(p[i] - 0.5) * 2.0),
            'apply': float(apply_eff[i]),
            'q_entropy': float(-np.sum(q[i] * np.log(np.clip(q[i], EPS, 1.0)))),
            'correction': float(z[i] - z0[i]),
            'best_candidate': int(best_candidate[i]),
            'best_expert': int(best_expert[i]),
            'oracle_gain_ce': float(ce_a[i] - oracle[i]),
            'final_oracle_gap_ce': float(ce_f[i] - oracle[i]),
            'high_conf_anchor_harm': int(abs(p0[i] - 0.5) * 2.0 > 0.8 and pred_a[i] == labels_np[i] and pred_f[i] != labels_np[i]),
        }
        for e in range(min(3, ze.shape[1])):
            row[f'p_expert{e}'] = float(pe[i, e])
            row[f'z_expert{e}'] = float(ze[i, e])
            row[f'pred_expert{e}'] = int(pred_e[i, e])
            row[f'expert{e}_correct'] = int(pred_e[i, e] == labels_np[i])
            row[f'ce_expert{e}'] = float(ce_e[i, e])
            row[f'q{e}'] = float(q[i, e])
            row[f'delta{e}'] = float(delta[i, e])
            row[f'delta_clamped{e}'] = float(delta_c[i, e])
        rows.append(row)
    return rows


def write_prediction_csv(args, rows, epoch, domain, degradation):
    if not rows or not getattr(args, 'dream_save_pred_csv', True) or not is_rank0():
        return
    ensure_output_dirs(args.output_dir)
    path = Path(args.output_dir) / 'predictions' / f'epoch_{epoch}_domain_{domain}_deg_{degradation}.csv'
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(json_safe(row))


def domain_metrics_from_rows(args, rows, epoch, domain, degradation):
    y = np.asarray([r['label'] for r in rows], dtype=np.float64)
    p_final = np.asarray([r['p_final'] for r in rows], dtype=np.float64)
    p_anchor = np.asarray([r['p_anchor'] for r in rows], dtype=np.float64)
    record = base_record(args, epoch, None, None, 'eval_domain', domain, degradation,
                         batch_size=len(rows))
    record['dream_fast_mode'] = getattr(args, 'dream_fast_mode', 'off')
    record['dream_fast_readout'] = getattr(args, 'dream_fast_readout', 'batchfold')
    mode = getattr(args, 'dream_fast_mode', 'off')
    if mode == 'single_bank':
        record['encoder_calls_expected'] = 1.0
        record['anchor_purity'] = 'bank_context'
    elif mode == 'bank_plus_anchor':
        record['encoder_calls_expected'] = 2.0
        record['anchor_purity'] = 'pure_anchor'
    else:
        record['encoder_calls_expected'] = float(getattr(args, 'dream_num_experts', 3) + 1)
        record['anchor_purity'] = 'pure_anchor'
    record['prompt_bank_len'] = (
        float((getattr(args, 'dream_num_experts', 3) + 1) * getattr(args, 'n_ctx', 2))
        if mode != 'off' else float(getattr(args, 'n_ctx', 2))
    )
    record.update(binary_metrics(y, p_final, 'final'))
    record.update(binary_metrics(y, p_anchor, 'anchor'))
    for e in range(3):
        key = f'p_expert{e}'
        if rows and key in rows[0]:
            record.update(binary_metrics(y, np.asarray([r[key] for r in rows], dtype=np.float64), f'expert{e}'))

    ce_f = np.asarray([r['ce_final'] for r in rows], dtype=np.float64)
    ce_a = np.asarray([r['ce_anchor'] for r in rows], dtype=np.float64)
    pred_f = np.asarray([r['pred_final'] for r in rows], dtype=bool)
    pred_a = np.asarray([r['pred_anchor'] for r in rows], dtype=bool)
    yb = y.astype(bool)
    f_ok = pred_f == yb
    a_ok = pred_a == yb
    record['acc_delta_final_anchor'] = record.get('final_acc', float('nan')) - record.get('anchor_acc', float('nan'))
    record['ap_delta_final_anchor'] = record.get('final_ap', float('nan')) - record.get('anchor_ap', float('nan'))
    record['racc_delta_final_anchor'] = record.get('final_racc', float('nan')) - record.get('anchor_racc', float('nan'))
    record['facc_delta_final_anchor'] = record.get('final_facc', float('nan')) - record.get('anchor_facc', float('nan'))
    record['brier_delta_final_anchor'] = record.get('final_brier', float('nan')) - record.get('anchor_brier', float('nan'))
    record['ece_delta_final_anchor'] = record.get('final_ece_10', float('nan')) - record.get('anchor_ece_10', float('nan'))
    cavr_eps = float(getattr(args, 'dream_cavr_eps', 1e-4))
    record['cavr_raw'] = float(np.mean(ce_f > ce_a))
    record['cavr_eps'] = float(np.mean(ce_f > ce_a + cavr_eps))
    record['cavr_strict'] = float(np.mean(ce_f > ce_a + 1e-3))
    record['cavr'] = record['cavr_eps']
    record['help_rate'] = float(np.mean(f_ok & ~a_ok))
    record['harm_rate'] = float(np.mean(~f_ok & a_ok))
    record['both_correct_rate'] = float(np.mean(f_ok & a_ok))
    record['both_wrong_rate'] = float(np.mean(~f_ok & ~a_ok))
    record['no_regret_index'] = record['help_rate'] - record['harm_rate']
    record['help_harm_ratio'] = record['help_rate'] / (record['harm_rate'] + EPS)
    flip = pred_f != pred_a
    record['decision_flip_rate'] = float(np.mean(flip))
    record['flip_help_rate'] = float(np.mean(flip & f_ok))
    record['flip_harm_rate'] = float(np.mean(flip & ~f_ok & a_ok))
    record['high_conf_anchor_harm_rate'] = safe_mean([r['high_conf_anchor_harm'] for r in rows])
    record['high_conf_anchor_flip_rate'] = float(np.mean((np.asarray([r['anchor_margin'] for r in rows]) > 0.8) & flip))
    add_distribution(record, 'correction_abs', np.abs(np.asarray([r['correction'] for r in rows])))
    record['correction_signed_mean'] = safe_mean([r['correction'] for r in rows])

    add_distribution(record, 'apply', [r['apply'] for r in rows])
    record['apply_rate_gt_005'] = safe_mean([r['apply'] > 0.05 for r in rows])
    record['apply_rate_gt_010'] = safe_mean([r['apply'] > 0.10 for r in rows])
    record['apply_rate_gt_050'] = safe_mean([r['apply'] > 0.50 for r in rows])
    q_cols = [f'q{e}' for e in range(3) if rows and f'q{e}' in rows[0]]
    if q_cols:
        q = np.asarray([[r[c] for c in q_cols] for r in rows], dtype=np.float64)
        q_entropy = -np.sum(q * np.log(np.clip(q, EPS, 1.0)), axis=1)
        q_top1 = np.argmax(q, axis=1)
        record['q_entropy_mean'] = float(q_entropy.mean())
        record['q_entropy_norm'] = float(q_entropy.mean() / math.log(max(q.shape[1], 2)))
        for e in range(q.shape[1]):
            record[f'q_mean_e{e}'] = float(q[:, e].mean())
            record[f'q_top1_rate_e{e}'] = float(np.mean(q_top1 == e))
            record[f'q_real_mean_e{e}'] = float(q[y == 0, e].mean()) if (y == 0).any() else float('nan')
            record[f'q_fake_mean_e{e}'] = float(q[y == 1, e].mean()) if (y == 1).any() else float('nan')
            record[f'q_anchor_correct_mean_e{e}'] = float(q[a_ok, e].mean()) if a_ok.any() else float('nan')
            record[f'q_anchor_wrong_mean_e{e}'] = float(q[~a_ok, e].mean()) if (~a_ok).any() else float('nan')
            record[f'delta_abs_e{e}'] = safe_mean([abs(r[f'delta{e}']) for r in rows])
            record[f'delta_signed_e{e}'] = safe_mean([r[f'delta{e}'] for r in rows])
            clip = float(getattr(args, 'dream_delta_clip', 1.0))
            record[f'delta_clip_frac_e{e}'] = safe_mean([abs(r[f'delta{e}']) >= clip - 1e-6 for r in rows])

    ce_experts = []
    for e in range(3):
        key = f'ce_expert{e}'
        if rows and key in rows[0]:
            cee = np.asarray([r[key] for r in rows], dtype=np.float64)
            ce_experts.append(cee)
            record[f'expert{e}_better_than_anchor_rate'] = float(np.mean(cee < ce_a))
    if ce_experts:
        ce_e = np.stack(ce_experts, axis=1)
        oracle_e = np.min(ce_e, axis=1)
        all_ce = np.concatenate([ce_a[:, None], ce_e], axis=1)
        oracle = np.min(all_ce, axis=1)
        best_c = np.argmin(all_ce, axis=1)
        best_e = np.argmin(ce_e, axis=1)
        record['expert_oracle_acc'] = safe_mean([
            rows[i][f'pred_expert{best_e[i]}'] == rows[i]['label'] for i in range(len(rows))
        ])
        record['candidate_oracle_acc'] = safe_mean([
            (rows[i]['pred_anchor'] if best_c[i] == 0 else rows[i][f'pred_expert{best_c[i] - 1}']) == rows[i]['label']
            for i in range(len(rows))
        ])
        record['expert_oracle_gain_ce'] = float(np.mean(ce_a - oracle_e))
        record['candidate_oracle_gain_ce'] = float(np.mean(ce_a - oracle))
        record['final_oracle_gap_ce'] = float(np.mean(ce_f - oracle))
        gain = ce_a - oracle
        mask = gain > 0.05
        capture = np.clip((ce_a - ce_f) / (gain + EPS), -5, 5)
        record['router_capture_rate'] = float(np.mean(capture[mask])) if mask.any() else float('nan')
        record['best_candidate_rate_anchor'] = float(np.mean(best_c == 0))
        for e in range(ce_e.shape[1]):
            record[f'best_candidate_rate_expert{e}'] = float(np.mean(best_c == e + 1))
            record[f'best_expert_rate_e{e}'] = float(np.mean(best_e == e))

    return record


def write_top_cases(args, rows, epoch, domain, degradation):
    if not rows or not is_rank0():
        return
    ensure_output_dirs(args.output_dir)
    def top(filter_fn, score_fn):
        cases = [r for r in rows if filter_fn(r)]
        cases = sorted(cases, key=score_fn, reverse=True)[:50]
        return cases
    payload = {
        'top_help_cases': top(lambda r: r['help'] == 1, lambda r: abs(r['correction']) + r['oracle_gain_ce']),
        'top_harm_cases': top(lambda r: r['harm'] == 1, lambda r: r['apply'] * abs(r['correction'])),
        'high_conf_anchor_harm_cases': top(lambda r: r['high_conf_anchor_harm'] == 1, lambda r: r['anchor_margin']),
        'high_apply_wrong_cases': top(lambda r: r['apply'] > 0.5 and not r['final_correct'], lambda r: r['apply']),
        'expert_oracle_missed_cases': top(lambda r: r['oracle_gain_ce'] > 0.05 and not r['final_correct'], lambda r: r['oracle_gain_ce']),
        'router_collapse_cases': top(lambda r: max([r.get('q0', 0), r.get('q1', 0), r.get('q2', 0)]) > 0.95 and not r['final_correct'], lambda r: max([r.get('q0', 0), r.get('q1', 0), r.get('q2', 0)])),
    }
    write_json(args.output_dir, f'cases/epoch_{epoch}_{domain}_{degradation}_top_cases.json', payload)


UFD_GROUPS = {
    'gan_group': ['progan', 'cyclegan', 'biggan', 'stylegan', 'gaugan', 'stargan'],
    'deepfake_group': ['deepfake'],
    'low_level_group': ['seeingdark', 'sitd', 'san'],
    'perceptual_loss_group': ['crn', 'imle'],
    'guided_group': ['guided'],
    'ldm_group': ['ldm_100', 'ldm_200', 'ldm_200_cfg'],
    'glide_group': ['glide_50_27', 'glide_100_10', 'glide_100_27'],
    'dalle_group': ['dalle'],
}


def mean_metrics_from_domains(args, domain_records, epoch, degradation):
    record = base_record(args, epoch, None, None, 'eval_mean', 'mean', degradation,
                         batch_size=sum(int(r.get('batch_size') or 0) for r in domain_records))
    if not domain_records:
        return record
    def vals(key):
        return [float(r[key]) for r in domain_records if key in r and isinstance(r[key], (int, float, np.floating, np.integer))]
    for prefix in ['final', 'anchor']:
        for metric in [
            'acc', 'ap', 'racc', 'facc', 'rf_gap', 'brier', 'ece_10',
            'ece_15', 'mce_10', 'nll', 'pos_rate', 'prob_mean',
            'prob_std', 'prob_real_mean', 'prob_fake_mean',
        ]:
            v = vals(f'{prefix}_{metric}')
            record[f'mean_{metric}_{prefix}'] = float(np.nanmean(v)) if v else float('nan')
            if metric == 'acc':
                record[f'std_acc_{prefix}'] = float(np.nanstd(v)) if v else float('nan')
                record[f'worst_acc_{prefix}'] = float(np.nanmin(v)) if v else float('nan')
                if v:
                    idx = int(np.nanargmin(v))
                    recs = [r for r in domain_records if f'{prefix}_{metric}' in r]
                    record[f'worst_domain_{prefix}'] = recs[idx]['domain']
                record[f'median_acc_{prefix}'] = float(np.nanmedian(v)) if v else float('nan')
    for key in ['acc_delta_final_anchor', 'ap_delta_final_anchor', 'cavr', 'help_rate', 'harm_rate',
                'no_regret_index', 'decision_flip_rate', 'high_conf_anchor_harm_rate',
                'apply_mean', 'q_entropy_mean', 'q_entropy_norm', 'correction_abs_mean',
                'expert_oracle_acc', 'candidate_oracle_acc', 'expert_oracle_gain_ce',
                'candidate_oracle_gain_ce', 'final_oracle_gap_ce', 'router_capture_rate']:
        v = vals(key)
        record[f'mean_{key}'] = float(np.nanmean(v)) if v else float('nan')
    for e in range(3):
        v = vals(f'expert{e}_better_than_anchor_rate')
        record[f'mean_expert{e}_better_than_anchor_rate'] = float(np.nanmean(v)) if v else float('nan')
    record['no_regression_domain_count_acc'] = int(sum(r.get('final_acc', 0) < r.get('anchor_acc', 0) for r in domain_records))
    record['no_regression_domain_names_acc'] = [r['domain'] for r in domain_records if r.get('final_acc', 0) < r.get('anchor_acc', 0)]
    record['no_regression_domain_count_ap'] = int(sum(r.get('final_ap', 0) < r.get('anchor_ap', 0) for r in domain_records))
    for e in range(3):
        for key in [f'q_mean_e{e}', f'q_top1_rate_e{e}', f'delta_abs_e{e}']:
            v = vals(key)
            record[f'mean_{key}'] = float(np.nanmean(v)) if v else float('nan')

    by_domain = {r['domain']: r for r in domain_records}
    for group, names in UFD_GROUPS.items():
        group_recs = [by_domain[n] for n in names if n in by_domain]
        if not group_recs:
            continue
        for metric in ['acc', 'ap']:
            record[f'{group}_{metric}_final'] = float(np.nanmean([r.get(f'final_{metric}', np.nan) for r in group_recs]))
            record[f'{group}_{metric}_anchor'] = float(np.nanmean([r.get(f'anchor_{metric}', np.nan) for r in group_recs]))
        record[f'{group}_acc_delta'] = record[f'{group}_acc_final'] - record[f'{group}_acc_anchor']
        record[f'{group}_cavr'] = float(np.nanmean([r.get('cavr', np.nan) for r in group_recs]))
        record[f'{group}_help'] = float(np.nanmean([r.get('help_rate', np.nan) for r in group_recs]))
        record[f'{group}_harm'] = float(np.nanmean([r.get('harm_rate', np.nan) for r in group_recs]))
        record[f'{group}_apply'] = float(np.nanmean([r.get('apply_mean', np.nan) for r in group_recs]))
        for e in range(3):
            record[f'{group}_q_mean_e{e}'] = float(np.nanmean([r.get(f'q_mean_e{e}', np.nan) for r in group_recs]))

    if 'progan' in by_domain:
        progan = by_domain['progan']
        unseen = [r for r in domain_records if r['domain'] != 'progan']
        record['seen_progan_acc_final'] = progan.get('final_acc', float('nan'))
        record['unseen_mean_acc_final'] = float(np.nanmean([r.get('final_acc', np.nan) for r in unseen])) if unseen else float('nan')
        record['seen_unseen_gap_final'] = record['seen_progan_acc_final'] - record['unseen_mean_acc_final']
        record['seen_unseen_gap_anchor'] = progan.get('anchor_acc', float('nan')) - (float(np.nanmean([r.get('anchor_acc', np.nan) for r in unseen])) if unseen else float('nan'))
        record['gap_delta'] = record['seen_unseen_gap_final'] - record['seen_unseen_gap_anchor']
    return record


def warning_flags(args, mean_record, epoch, degradation='none'):
    flags = []
    def add(name, cond, message):
        if cond:
            flags.append({
                **base_record(args, epoch, None, None, 'warning', 'mean', degradation),
                'warning': name,
                'message': message,
            })
    add('ap_high_acc_low', mean_record.get('mean_ap_final', 0) > 0.95 and mean_record.get('mean_acc_final', 1) < 0.85,
        'AP high but fixed-threshold Acc low.')
    add('real_fake_collapse', mean_record.get('mean_racc_final', 1) < 0.2 or mean_record.get('mean_facc_final', 1) < 0.2,
        'Real or fake accuracy collapsed.')
    add('over_correction', mean_record.get('mean_harm_rate', 0) > mean_record.get('mean_help_rate', 0),
        'Harm rate is higher than help rate.')
    add('anchor_damage', mean_record.get('mean_cavr', 0) > 0.45,
        'Final CE is worse than anchor CE on too many samples.')
    add('expert_dead', mean_record.get('mean_apply_mean', 1) < 0.01 and epoch >= getattr(args, 'dream_warmup_epochs', 1) + 1,
        'Apply probability is near zero after warmup.')
    qent = mean_record.get('mean_q_entropy_mean', mean_record.get('mean_q_entropy_norm', 1))
    add('router_collapse', mean_record.get('mean_q_entropy_norm', qent) < 0.2 and epoch >= 2,
        'Router entropy is very low.')
    add('seen_overfit', mean_record.get('seen_unseen_gap_final', -999) > mean_record.get('seen_unseen_gap_anchor', 999) + 0.05,
        'Seen-unseen gap grew relative to anchor.')
    add('calibration_shift', abs(mean_record.get('mean_racc_final', 0) - mean_record.get('mean_facc_final', 0)) >
        abs(mean_record.get('mean_racc_anchor', 0) - mean_record.get('mean_facc_anchor', 0)) + 0.1,
        'RAcc/FAcc imbalance worsened relative to anchor.')
    clip_vals = [mean_record.get(f'mean_delta_clip_frac_e{e}', 0) for e in range(3)]
    add('clip_saturation', any(v > 0.5 for v in clip_vals if isinstance(v, (int, float))),
        'More than half of deltas hit the correction clamp for an expert.')
    for flag in flags:
        write_jsonl(args.output_dir, 'warning_flags.jsonl', flag)
        print('WARNING [{}] {}'.format(flag['warning'], flag['message']))
    return flags


def write_startup_sanity(args, model):
    if getattr(args, 'method', 'iapl') != 'dream_cs' or not is_rank0():
        return
    ensure_output_dirs(args.output_dir)
    checks = {
        'method_is_dream_cs': getattr(args, 'method', None) == 'dream_cs',
        'standalone_no_dream_anchor_ckpt': getattr(args, 'dream_anchor_ckpt', '') == '',
        'dream_freeze_anchor_false': not getattr(args, 'dream_freeze_anchor', False),
        'dream_residual_scale_init_positive': getattr(args, 'dream_residual_scale_init', 0.0) > 0,
        'clean_safe_uses_detached_anchor_reference': True,
        'route_target_detaches_anchor_and_expert_ce': True,
        'experts_share_fc_binary_no_independent_head': not any('expert' in n.lower() and 'head' in n.lower() for n, _ in model.named_modules()),
        'router_low_dim_statistics_only': True,
        'tta_false_for_training': not getattr(args, 'tta', False),
        'label_convention': 'real=0, fake=1; RAcc uses label==0 and FAcc uses label==1',
    }
    warnings = []
    if not checks['standalone_no_dream_anchor_ckpt']:
        warnings.append('WARNING: dream_anchor_ckpt is set. This is plugin/warmstart ablation, not standalone main setting.')
    if not checks['dream_freeze_anchor_false']:
        warnings.append('WARNING: dream_freeze_anchor=True. This is not standalone joint training.')
    if not checks['dream_residual_scale_init_positive']:
        raise ValueError('dream_residual_scale_init must be > 0 for DREAM-CS Standalone.')
    if getattr(args, 'tta', False):
        warnings.append('WARNING: args.tta=True. Current request is no-TTA training.')
    config = vars(args).copy()
    config['sanity_checks'] = checks
    config['sanity_warnings'] = warnings
    config['label_convention'] = checks['label_convention']
    write_json(args.output_dir, 'config_snapshot.json', config)
    text = ['DREAM-CS Standalone sanity check', '']
    for key, value in checks.items():
        text.append(f'{key}: {value}')
    if warnings:
        text.extend(['', 'Warnings:'])
        text.extend(warnings)
    text.append('')
    text.append('Train degradation uses GPU tensor jpeg-like/webp-like approximation, not real codecs.')
    Path(args.output_dir, 'sanity_check.txt').write_text('\n'.join(text))
    for warning in warnings:
        print(warning)


def write_corruption_summary(args, per_deg_mean, epoch):
    if not per_deg_mean or not is_rank0():
        return
    clean = per_deg_mean.get('none')
    corrupt = {k: v for k, v in per_deg_mean.items() if k != 'none'}
    for name, rec in per_deg_mean.items():
        row = base_record(args, epoch, None, None, 'eval_corruption', 'mean', name)
        row.update({
            'deg_name': name,
            'mean_acc_final': rec.get('mean_acc_final'),
            'mean_ap_final': rec.get('mean_ap_final'),
            'mean_acc_anchor': rec.get('mean_acc_anchor'),
            'mean_ap_anchor': rec.get('mean_ap_anchor'),
            'mean_acc_delta_final_anchor': rec.get('mean_acc_delta_final_anchor'),
            'mean_cavr': rec.get('mean_cavr'),
            'mean_help': rec.get('mean_help_rate'),
            'mean_harm': rec.get('mean_harm_rate'),
            'mean_apply': rec.get('mean_apply_mean'),
            'mean_q_entropy': rec.get('mean_q_entropy_mean'),
            'mean_correction_abs': rec.get('mean_correction_abs_mean'),
            'mean_rf_gap_final': rec.get('mean_rf_gap_final'),
            'worst_domain': rec.get('worst_domain_final'),
            'worst_domain_acc_final': rec.get('worst_acc_final'),
        })
        for e in range(3):
            row[f'mean_q_e{e}'] = rec.get(f'mean_q_mean_e{e}')
        if clean and name != 'none':
            row['acc_drop_final'] = clean.get('mean_acc_final', np.nan) - rec.get('mean_acc_final', np.nan)
            row['acc_drop_anchor'] = clean.get('mean_acc_anchor', np.nan) - rec.get('mean_acc_anchor', np.nan)
            row['drop_reduction'] = row['acc_drop_anchor'] - row['acc_drop_final']
            row['ap_drop_final'] = clean.get('mean_ap_final', np.nan) - rec.get('mean_ap_final', np.nan)
            row['ap_drop_anchor'] = clean.get('mean_ap_anchor', np.nan) - rec.get('mean_ap_anchor', np.nan)
            row[f'apply_shift_{name}'] = rec.get('mean_apply_mean', np.nan) - clean.get('mean_apply_mean', np.nan)
            for e in range(3):
                row[f'q_shift_{name}_e{e}'] = rec.get(f'mean_q_mean_e{e}', np.nan) - clean.get(f'mean_q_mean_e{e}', np.nan)
        if corrupt:
            row['worst_corrupt_acc_final'] = float(np.nanmin([v.get('mean_acc_final', np.nan) for v in corrupt.values()]))
            row['worst_corrupt_acc_anchor'] = float(np.nanmin([v.get('mean_acc_anchor', np.nan) for v in corrupt.values()]))
            row['worst_corrupt_improvement'] = row['worst_corrupt_acc_final'] - row['worst_corrupt_acc_anchor']
            row['avg_corrupt_acc_final'] = float(np.nanmean([v.get('mean_acc_final', np.nan) for v in corrupt.values()]))
            row['avg_corrupt_acc_anchor'] = float(np.nanmean([v.get('mean_acc_anchor', np.nan) for v in corrupt.values()]))
            if clean:
                row['avg_corrupt_drop_final'] = clean.get('mean_acc_final', np.nan) - row['avg_corrupt_acc_final']
                row['avg_corrupt_drop_anchor'] = clean.get('mean_acc_anchor', np.nan) - row['avg_corrupt_acc_anchor']
        write_jsonl(args.output_dir, 'eval_corruption_summary.jsonl', row)


def write_diagnosis_summary(output_dir, args):
    if not output_dir or not is_rank0():
        return
    root = Path(output_dir)
    mean_path = root / 'eval_mean_metrics.jsonl'
    if not mean_path.exists():
        return
    records = []
    with mean_path.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        return
    last = records[-1]
    best = max(records, key=lambda r: r.get('mean_acc_final', float('-inf')) if isinstance(r.get('mean_acc_final'), (int, float)) else float('-inf'))
    warnings = []
    warn_path = root / 'warning_flags.jsonl'
    if warn_path.exists():
        with warn_path.open() as f:
            warnings = [json.loads(line) for line in f if line.strip()]
    recs = []
    if last.get('mean_apply_mean', 1) < 0.01:
        recs.append('Expert dead: raise loss_dream_expert or delay dream_router_start_epoch.')
    if last.get('mean_q_entropy_norm', 1) < 0.2:
        recs.append('Router collapse: reduce loss_dream_route or raise loss_dream_div/loss_dream_specialize.')
    if last.get('mean_harm_rate', 0) > last.get('mean_help_rate', 0):
        recs.append('Harm > help: raise clean-safe, lower delta_clip, or make apply bias more negative.')
    if last.get('mean_ap_final', 0) > 0.95 and last.get('mean_acc_final', 1) < 0.85:
        recs.append('AP high but Acc low: inspect pos_rate/rf_gap and calibration threshold drift.')
    if not recs:
        recs.append('No automatic red flag dominates; inspect per-domain CSV/case files next.')
    lines = [
        '# DREAM-CS Standalone Quick Diagnosis',
        '',
        '## Best / Last Final vs Anchor',
        f"best_epoch={best.get('epoch')} best_final_acc={best.get('mean_acc_final')} best_final_ap={best.get('mean_ap_final')} anchor_acc={best.get('mean_acc_anchor')} anchor_ap={best.get('mean_ap_anchor')}",
        f"last_epoch={last.get('epoch')} final_acc={last.get('mean_acc_final')} final_ap={last.get('mean_ap_final')} racc={last.get('mean_racc_final')} facc={last.get('mean_facc_final')} rf_gap={last.get('mean_rf_gap_final')} ece={last.get('mean_ece_10_final')} brier={last.get('mean_brier_final')}",
        '',
        '## No-Regret',
        f"help={last.get('mean_help_rate')} harm={last.get('mean_harm_rate')} CAVR={last.get('mean_cavr')} no_regret={last.get('mean_no_regret_index')} high_conf_anchor_harm={last.get('mean_high_conf_anchor_harm_rate')}",
        '',
        '## Router',
        f"apply={last.get('mean_apply_mean')} q_entropy={last.get('mean_q_entropy_mean')} q={[last.get(f'mean_q_mean_e{i}') for i in range(3)]} q_top1={[last.get(f'mean_q_top1_rate_e{i}') for i in range(3)]}",
        '',
        '## Experts',
        f"expert_oracle_gain={last.get('mean_expert_oracle_gain_ce')} final_oracle_gap={last.get('mean_final_oracle_gap_ce')} expert_better_rates={[last.get(f'mean_expert{i}_better_than_anchor_rate') for i in range(3)]}",
        '',
        '## Red Flags',
    ]
    if warnings:
        lines.extend([f"- epoch={w.get('epoch')} deg={w.get('degradation')} {w.get('warning')}: {w.get('message')}" for w in warnings])
    else:
        lines.append('- none recorded')
    lines.extend(['', '## Recommendation'])
    lines.extend([f'- {r}' for r in recs])
    (root / 'diagnosis_summary.md').write_text('\n'.join(lines) + '\n')
