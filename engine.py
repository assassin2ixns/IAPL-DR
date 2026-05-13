import math
import sys
from typing import Iterable
import torch
import torch.nn.functional as F
import utils.misc as utils
import numpy as np
import torch.distributed as dist
from sklearn.metrics import average_precision_score, accuracy_score


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


def grad_norm_named(model, include_substr):
    total = None
    for name, param in model.named_parameters():
        if include_substr in name and param.grad is not None:
            value = param.grad.detach().float().norm(2).pow(2)
            total = value if total is None else total + value
    if total is None:
        return 0.0
    return total.sqrt()


def _safe_mean(tensor):
    if tensor.numel() == 0:
        return None
    return tensor.float().mean()


def _safe_acc(probs, labels):
    if probs.numel() == 0:
        return None
    return ((probs > 0.5).float() == labels.float()).float().mean()


def _binary_metrics(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    pred = y_prob > 0.5
    real_mask = y_true == 0
    fake_mask = y_true == 1
    metrics = {
        'acc': accuracy_score(y_true, pred),
        'ap': average_precision_score(y_true, y_prob),
        'racc': accuracy_score(y_true[real_mask], pred[real_mask]) if real_mask.any() else np.nan,
        'facc': accuracy_score(y_true[fake_mask], pred[fake_mask]) if fake_mask.any() else np.nan,
    }
    return metrics


def _brier(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    return float(np.mean((y_prob - y_true) ** 2))


def _ece(y_true, y_prob, bins=10):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    conf = np.maximum(y_prob, 1.0 - y_prob)
    pred = (y_prob > 0.5).astype(np.float32)
    correct = (pred == y_true).astype(np.float32)
    ece = 0.0
    for low in np.linspace(0.0, 1.0, bins, endpoint=False):
        high = low + 1.0 / bins
        mask = (conf >= low) & (conf < high if high < 1.0 else conf <= high)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return ece


def _update_if_not_none(metric_logger, **kwargs):
    clean = {k: v for k, v in kwargs.items() if v is not None}
    if clean:
        metric_logger.update(**clean)


def _log_dream_train_metrics(metric_logger, outputs, labels, loss_dict, model_without_ddp):
    logits = outputs['logits_flat'].detach()
    labels_f = labels.float().view(-1)
    probs = logits.sigmoid()
    anchor_probs = outputs['anchor_logits'].detach().sigmoid()
    anchor_logits = outputs['anchor_logits'].detach()
    q = outputs['q'].detach()
    apply_eff = outputs['apply'].detach()
    apply_raw = outputs.get('apply_raw', apply_eff).detach()
    delta = outputs.get('delta_raw', outputs['expert_logits'].detach() - anchor_logits[:, None]).detach()
    delta_clamped = outputs.get('delta_clamped', delta).detach()
    experts = outputs['expert_logits'].detach().sigmoid()

    real_mask = labels_f == 0
    fake_mask = labels_f == 1
    final_acc = _safe_acc(probs, labels_f)
    anchor_acc = _safe_acc(anchor_probs, labels_f)
    racc_final = _safe_acc(probs[real_mask], labels_f[real_mask])
    facc_final = _safe_acc(probs[fake_mask], labels_f[fake_mask])
    racc_anchor = _safe_acc(anchor_probs[real_mask], labels_f[real_mask])
    facc_anchor = _safe_acc(anchor_probs[fake_mask], labels_f[fake_mask])
    rf_gap = None
    if racc_final is not None and facc_final is not None:
        rf_gap = (racc_final - facc_final).abs()

    entropy = -(q.clamp_min(1e-8) * q.clamp_min(1e-8).log()).sum(dim=1).mean()
    top1 = q.argmax(dim=1)
    correction = logits - anchor_logits
    _update_if_not_none(
        metric_logger,
        dream_prob_mean=probs.mean(),
        dream_prob_std=probs.std(unbiased=False),
        dream_pos_rate=(probs > 0.5).float().mean(),
        dream_label_pos_rate=labels_f.mean(),
        dream_anchor_prob_mean=anchor_probs.mean(),
        dream_anchor_pos_rate=(anchor_probs > 0.5).float().mean(),
        dream_final_anchor_prob_gap=(probs - anchor_probs).abs().mean(),
        dream_train_acc_final=final_acc,
        dream_train_acc_anchor=anchor_acc,
        dream_train_racc_final=racc_final,
        dream_train_facc_final=facc_final,
        dream_train_racc_anchor=racc_anchor,
        dream_train_facc_anchor=facc_anchor,
        dream_real_prob_final_mean=_safe_mean(probs[real_mask]),
        dream_fake_prob_final_mean=_safe_mean(probs[fake_mask]),
        dream_real_prob_anchor_mean=_safe_mean(anchor_probs[real_mask]),
        dream_fake_prob_anchor_mean=_safe_mean(anchor_probs[fake_mask]),
        dream_batch_rf_gap=rf_gap,
        dream_apply_mean=apply_eff.mean(),
        dream_apply_raw_mean=apply_raw.mean(),
        dream_apply_std=apply_eff.std(unbiased=False),
        dream_apply_rate_05=(apply_eff > 0.5).float().mean(),
        dream_q_entropy=entropy,
        dream_correction_abs=correction.abs().mean(),
        dream_correction_signed=correction.mean(),
        dream_residual_norm=outputs['prompt_residual_norm'].detach(),
        dream_residual_cos_offdiag=outputs.get('residual_cos_offdiag', logits.sum() * 0.0).detach(),
    )

    for idx in range(min(3, q.shape[1])):
        e_mask = top1 == idx
        _update_if_not_none(
            metric_logger,
            **{
                'dream_q_mean_e{}'.format(idx): q[:, idx].mean(),
                'dream_q_top1_e{}'.format(idx): e_mask.float().mean(),
                'dream_q_real_e{}'.format(idx): _safe_mean(q[real_mask, idx]),
                'dream_q_fake_e{}'.format(idx): _safe_mean(q[fake_mask, idx]),
                'dream_delta_abs_e{}'.format(idx): delta[:, idx].abs().mean(),
                'dream_delta_signed_e{}'.format(idx): delta[:, idx].mean(),
                'dream_delta_clip_frac_e{}'.format(idx): (delta[:, idx] != delta_clamped[:, idx]).float().mean(),
                'dream_expert_prob_mean_e{}'.format(idx): experts[:, idx].mean(),
                'dream_expert_acc_e{}'.format(idx): _safe_acc(experts[:, idx], labels_f),
                'dream_feature_res_norm_e{}'.format(idx): outputs['feature_residual_norm_per_expert'][:, idx].detach().mean(),
                'dream_residual_norm_e{}'.format(idx): outputs['residual_norm_per_expert'][idx].detach(),
                'dream_residual_scale_e{}'.format(idx): outputs['residual_scale_per_expert'][idx].detach(),
            }
        )

    for key, value in loss_dict.items():
        if key.startswith('stat_'):
            metric_logger.update(**{'dream_' + key[len('stat_'):]: value.detach()})

    deg_outputs = outputs.get('deg_outputs', [])
    if len(deg_outputs) > 0:
        stack = torch.stack([outputs['logits_flat'].detach()] + [d['logits_flat'].detach() for d in deg_outputs], dim=0)
        deg_final_ce = []
        deg_anchor_ce = []
        counts = {name: 0 for name in ['jpeg90', 'jpeg75', 'jpeg50', 'resize', 'blur', 'quant', 'webp']}
        for deg_out in deg_outputs:
            name = deg_out.get('deg_name', 'unknown')
            if name in counts:
                counts[name] += 1
            deg_final_ce.append(F.binary_cross_entropy_with_logits(deg_out['logits_flat'], labels_f, reduction='none').mean())
            deg_anchor_ce.append(F.binary_cross_entropy_with_logits(deg_out['anchor_logits'], labels_f, reduction='none').mean())
        metric_logger.update(
            dream_deg_num_views=float(len(deg_outputs)),
            dream_logit_var_clean_deg=stack.var(dim=0, unbiased=False).mean(),
            dream_deg_final_ce_mean=torch.stack(deg_final_ce).mean(),
            dream_deg_anchor_ce_mean=torch.stack(deg_anchor_ce).mean(),
        )
        for name, count in counts.items():
            metric_logger.update(**{'dream_deg_names_count_' + name: float(count) / float(len(deg_outputs))})

    metric_logger.update(
        grad_norm_dream_router=grad_norm_named(model_without_ddp, 'dream_router'),
        grad_norm_dream_expert_bank=grad_norm_named(model_without_ddp, 'dream_expert_bank'),
        grad_norm_dream_expert_scale=grad_norm_named(model_without_ddp, 'dream_expert_bank.scale'),
        grad_norm_anchor_prompt=grad_norm_named(model_without_ddp, 'prompt_learner'),
        grad_norm_fc_binary=grad_norm_named(model_without_ddp, 'fc_binary'),
    )


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    epoch: int, lr_scheduler=None, max_norm: float = 0, args=None, model_ema=None):

    model.train()
    model_without_ddp = unwrap_model(model)
    if hasattr(model_without_ddp, 'set_epoch'):
        model_without_ddp.set_epoch(epoch)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = args.print_freq
    for samples in metric_logger.log_every(data_loader, print_freq, header):
        images, labels = [sample.to(device) for sample in samples]
        outputs = model(images)
        loss_dict = model_without_ddp.get_criterion(outputs, labels)
        weight_dict = model_without_ddp.criterion_weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if isinstance(outputs, dict) and getattr(args, 'method', 'iapl') == 'dream_cs' and getattr(args, 'dream_log_router', True):
            _log_dream_train_metrics(metric_logger, outputs, labels, loss_dict, model_without_ddp)
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        if model_ema:
            model_ema.update(model)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)


@torch.no_grad()
def gather_together(data):
    world_size = utils.get_world_size()
    if world_size < 2:
        return data
    dist.barrier()
    gather_data = [None for _ in range(world_size)]
    dist.all_gather_object(gather_data, data)
    return gather_data


def _print_dream_eval(data_name, y_true, p_final, p_anchor, p_experts, apply_vals, q_vals, delta_vals, clip_vals, corr_vals):
    final_m = _binary_metrics(y_true, p_final)
    anchor_m = _binary_metrics(y_true, p_anchor)
    y_true_np = np.asarray(y_true)
    final_pred = np.asarray(p_final) > 0.5
    anchor_pred = np.asarray(p_anchor) > 0.5
    final_correct = final_pred == y_true_np
    anchor_correct = anchor_pred == y_true_np
    final_ce = -(y_true_np * np.log(np.asarray(p_final) + 1e-8) + (1 - y_true_np) * np.log(1 - np.asarray(p_final) + 1e-8))
    anchor_ce = -(y_true_np * np.log(np.asarray(p_anchor) + 1e-8) + (1 - y_true_np) * np.log(1 - np.asarray(p_anchor) + 1e-8))
    cavr = float(np.mean(final_ce > anchor_ce))
    help_rate = float(np.mean(final_correct & ~anchor_correct))
    harm_rate = float(np.mean(~final_correct & anchor_correct))
    both_wrong = float(np.mean(~final_correct & ~anchor_correct))
    both_correct = float(np.mean(final_correct & anchor_correct))
    real_mask = y_true_np == 0
    fake_mask = y_true_np == 1
    q_arr = np.asarray(q_vals) if len(q_vals) else np.zeros((0, 0))
    q_mean = q_arr.mean(axis=0) if q_arr.size else np.zeros(0)
    q_top1 = np.bincount(q_arr.argmax(axis=1), minlength=q_arr.shape[1]) / max(1, q_arr.shape[0]) if q_arr.size else np.zeros(0)
    q_entropy = float(np.mean(-np.sum(q_arr * np.log(np.clip(q_arr, 1e-8, 1.0)), axis=1))) if q_arr.size else 0.0
    delta_arr = np.asarray(delta_vals) if len(delta_vals) else np.zeros((0, 0))
    clip_arr = np.asarray(clip_vals) if len(clip_vals) else np.zeros((0, 0))
    corr_arr = np.asarray(corr_vals) if len(corr_vals) else np.zeros(0)

    print("[DREAM-EVAL] domain={} final acc={:.2f} ap={:.2f} racc={:.2f} facc={:.2f}".format(
        data_name, final_m['acc'] * 100, final_m['ap'] * 100, final_m['racc'] * 100, final_m['facc'] * 100))
    print("[DREAM-EVAL] domain={} anchor acc={:.2f} ap={:.2f} racc={:.2f} facc={:.2f}".format(
        data_name, anchor_m['acc'] * 100, anchor_m['ap'] * 100, anchor_m['racc'] * 100, anchor_m['facc'] * 100))
    for idx, probs in enumerate(p_experts):
        expert_m = _binary_metrics(y_true, probs)
        print("[DREAM-EVAL] domain={} expert{} acc={:.2f} ap={:.2f} racc={:.2f} facc={:.2f}".format(
            data_name, idx, expert_m['acc'] * 100, expert_m['ap'] * 100, expert_m['racc'] * 100, expert_m['facc'] * 100))
    print("[DREAM-EVAL] domain={} cavr={:.4f} help={:.4f} harm={:.4f} both_wrong={:.4f} both_correct={:.4f} apply={:.4f} q={} q_top1={} q_entropy={:.4f} rf_gap={:.4f} ece={:.4f}".format(
        data_name,
        cavr,
        help_rate,
        harm_rate,
        both_wrong,
        both_correct,
        float(np.mean(apply_vals)) if len(apply_vals) else 0.0,
        np.round(q_mean, 4).tolist(),
        np.round(q_top1, 4).tolist(),
        q_entropy,
        abs(final_m['racc'] - final_m['facc']),
        _ece(y_true, p_final),
    ))
    print("[DREAM-EVAL] domain={} no_regret acc_delta={:.2f} ap_delta={:.2f} pos_final={:.4f} pos_anchor={:.4f} brier_final={:.4f} brier_anchor={:.4f} prob_real_final={:.4f} prob_fake_final={:.4f} prob_real_anchor={:.4f} prob_fake_anchor={:.4f} correction_abs={:.4f} delta_abs={} clip_frac={}".format(
        data_name,
        (final_m['acc'] - anchor_m['acc']) * 100,
        (final_m['ap'] - anchor_m['ap']) * 100,
        float(np.mean(np.asarray(p_final) > 0.5)),
        float(np.mean(np.asarray(p_anchor) > 0.5)),
        _brier(y_true, p_final),
        _brier(y_true, p_anchor),
        float(np.mean(np.asarray(p_final)[real_mask])) if real_mask.any() else float('nan'),
        float(np.mean(np.asarray(p_final)[fake_mask])) if fake_mask.any() else float('nan'),
        float(np.mean(np.asarray(p_anchor)[real_mask])) if real_mask.any() else float('nan'),
        float(np.mean(np.asarray(p_anchor)[fake_mask])) if fake_mask.any() else float('nan'),
        float(np.mean(np.abs(corr_arr))) if corr_arr.size else 0.0,
        np.round(np.mean(np.abs(delta_arr), axis=0), 4).tolist() if delta_arr.size else [],
        np.round(np.mean(clip_arr, axis=0), 4).tolist() if clip_arr.size else [],
    ))
    return {
        'final': final_m,
        'anchor': anchor_m,
        'cavr': cavr,
        'help': help_rate,
        'harm': harm_rate,
        'apply': float(np.mean(apply_vals)) if len(apply_vals) else 0.0,
        'q_mean': q_mean,
    }


@torch.no_grad()
def _evaluate_once(model, data_loaders, device, args=None, degradation_name=None):
    model.eval()
    test_dataset = []
    test_AP = []
    test_ACC = []
    test_real_ACC = []
    test_fake_ACC = []
    dream_summaries = []
    if degradation_name is not None:
        print('[EVAL-DEG] degradation={}'.format(degradation_name))
        old_deg = getattr(args, 'dream_eval_degradation', 'none')
        args.dream_eval_degradation = degradation_name
    elif (
        getattr(args, 'method', 'iapl') == 'dream_cs'
        and getattr(args, 'dream_eval_degradation', 'none') != 'none'
    ):
        print('DREAM-CS eval degradation: {}'.format(args.dream_eval_degradation))

    for data_name, data_loader in data_loaders.items():
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Test:'
        print_freq = args.print_freq

        y_true, y_pred = [], []
        p_anchor, apply_vals, corr_vals = [], [], []
        q_vals, delta_vals, clip_vals = [], [], []
        p_experts = None

        for batch_idx, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            if getattr(args, 'eval_max_batches_per_domain', 0) and batch_idx >= args.eval_max_batches_per_domain:
                break
            images, labels = [sample.to(device) for sample in samples]
            outputs = model(images)
            logits = extract_logits(outputs)
            probs = logits.sigmoid().flatten()
            y_pred.extend(probs.tolist())
            y_true.extend(labels.flatten().tolist())
            if isinstance(outputs, dict) and getattr(args, 'method', 'iapl') == 'dream_cs':
                p_anchor.extend(outputs['anchor_logits'].sigmoid().flatten().tolist())
                apply_vals.extend(outputs['apply'].flatten().tolist())
                q_vals.extend(outputs['q'].detach().cpu().tolist())
                delta_vals.extend(outputs['delta_raw'].detach().cpu().tolist())
                clip_vals.extend((outputs['delta_raw'] != outputs['delta_clamped']).float().detach().cpu().tolist())
                corr_vals.extend((outputs['logits_flat'] - outputs['anchor_logits']).detach().cpu().flatten().tolist())
                expert_probs = outputs['expert_logits'].sigmoid().detach().cpu()
                if p_experts is None:
                    p_experts = [[] for _ in range(expert_probs.shape[1])]
                for idx in range(expert_probs.shape[1]):
                    p_experts[idx].extend(expert_probs[:, idx].tolist())

        world_size = utils.get_world_size()
        if world_size < 2:
            merge_y_true = y_true
            merge_y_pred = y_pred
            merge_anchor = p_anchor
            merge_apply = apply_vals
            merge_q = q_vals
            merge_delta = delta_vals
            merge_clip = clip_vals
            merge_corr = corr_vals
            merge_experts = p_experts
        else:
            merge_y_true, merge_y_pred = [], []
            for data in gather_together(y_true):
                merge_y_true.extend(data)
            for data in gather_together(y_pred):
                merge_y_pred.extend(data)
            merge_anchor, merge_apply, merge_q, merge_delta, merge_clip, merge_corr = [], [], [], [], [], []
            for data in gather_together(p_anchor):
                merge_anchor.extend(data)
            for data in gather_together(apply_vals):
                merge_apply.extend(data)
            for data in gather_together(q_vals):
                merge_q.extend(data)
            for data in gather_together(delta_vals):
                merge_delta.extend(data)
            for data in gather_together(clip_vals):
                merge_clip.extend(data)
            for data in gather_together(corr_vals):
                merge_corr.extend(data)
            merge_experts = None
            if p_experts is not None:
                gathered = gather_together(p_experts)
                merge_experts = [[] for _ in range(len(p_experts))]
                for rank_experts in gathered:
                    for idx in range(len(merge_experts)):
                        merge_experts[idx].extend(rank_experts[idx])

        y_true_np, y_pred_np = np.array(merge_y_true), np.array(merge_y_pred)
        metrics = _binary_metrics(y_true_np, y_pred_np)

        test_dataset.append(data_name)
        test_AP.append(metrics['ap'])
        test_ACC.append(metrics['acc'])
        test_real_ACC.append(metrics['racc'])
        test_fake_ACC.append(metrics['facc'])

        print("({}) acc: {:.2f}; ap: {:.2f}; racc: {:.2f}; facc: {:.2f};".format(
            data_name, metrics['acc'] * 100, metrics['ap'] * 100, metrics['racc'] * 100, metrics['facc'] * 100))

        if merge_experts is not None and len(merge_anchor) == len(merge_y_true):
            dream_summaries.append(_print_dream_eval(
                data_name, merge_y_true, merge_y_pred, merge_anchor, merge_experts,
                merge_apply, merge_q, merge_delta, merge_clip, merge_corr))

    output_strs = []
    for idx, [name, ap, acc, racc, facc] in enumerate(zip(
        test_dataset + ["mean"],
        test_AP + [np.nanmean(test_AP)],
        test_ACC + [np.nanmean(test_ACC)],
        test_real_ACC + [np.nanmean(test_real_ACC)],
        test_fake_ACC + [np.nanmean(test_fake_ACC)],
    )):
        output_str = "({} {:10}) acc: {:.2f}; ap: {:.2f}; racc: {:.2f}; facc: {:.2f};".format(
            idx, name, acc * 100, ap * 100, racc * 100, facc * 100)
        output_strs.append(output_str)
        print(output_str)

    if dream_summaries:
        mean_anchor_acc = np.mean([s['anchor']['acc'] for s in dream_summaries])
        mean_anchor_ap = np.mean([s['anchor']['ap'] for s in dream_summaries])
        mean_cavr = np.mean([s['cavr'] for s in dream_summaries])
        mean_help = np.mean([s['help'] for s in dream_summaries])
        mean_harm = np.mean([s['harm'] for s in dream_summaries])
        mean_apply = np.mean([s['apply'] for s in dream_summaries])
        q_stack = [s['q_mean'] for s in dream_summaries if len(s['q_mean']) > 0]
        q_mean = np.mean(np.stack(q_stack, axis=0), axis=0) if q_stack else []
        print("[DREAM-EVAL-MEAN] anchor_acc={:.2f} anchor_ap={:.2f} cavr={:.4f} help={:.4f} harm={:.4f} apply={:.4f} q={}".format(
            mean_anchor_acc * 100, mean_anchor_ap * 100, mean_cavr, mean_help, mean_harm, mean_apply,
            np.round(q_mean, 4).tolist() if len(q_mean) else []))
        output_strs.append("[DREAM mean anchor_acc {:.2f} anchor_ap {:.2f} cavr {:.4f} help {:.4f} harm {:.4f} apply {:.4f}]".format(
            mean_anchor_acc * 100, mean_anchor_ap * 100, mean_cavr, mean_help, mean_harm, mean_apply))

    if degradation_name is not None:
        args.dream_eval_degradation = old_deg
    return "; ".join(output_strs), np.nanmean(test_AP), np.nanmean(test_ACC), {
        'ap': np.nanmean(test_AP),
        'acc': np.nanmean(test_ACC),
    }


@torch.no_grad()
def evaluate(model, data_loaders, device, args=None, test=False):
    multi = getattr(args, 'dream_eval_multi_degradations', [])
    if getattr(args, 'method', 'iapl') == 'dream_cs' and len(multi) > 0:
        outputs = []
        results = {}
        for name in multi:
            output_strs, cur_ap, cur_acc, summary = _evaluate_once(model, data_loaders, device, args=args, degradation_name=name)
            outputs.append('[{}] {}'.format(name, output_strs))
            results[name] = summary
        if 'none' in results:
            clean_acc = results['none']['acc']
            clean_ap = results['none']['ap']
            corrupt = {k: v for k, v in results.items() if k != 'none'}
            if corrupt:
                worst_acc = min(v['acc'] for v in corrupt.values())
                worst_ap = min(v['ap'] for v in corrupt.values())
                avg_drop = np.mean([clean_acc - v['acc'] for v in corrupt.values()])
                print("[DREAM-EVAL-MULTI] clean_acc={:.2f} clean_ap={:.2f} worst_corrupt_acc={:.2f} worst_corrupt_ap={:.2f} avg_acc_drop={:.2f}".format(
                    clean_acc * 100, clean_ap * 100, worst_acc * 100, worst_ap * 100, avg_drop * 100))
        mean_ap = results[multi[0]]['ap']
        mean_acc = results[multi[0]]['acc']
        return " | ".join(outputs), mean_ap, mean_acc

    output_strs, cur_ap, cur_acc, _ = _evaluate_once(model, data_loaders, device, args=args)
    return output_strs, cur_ap, cur_acc
