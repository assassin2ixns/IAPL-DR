import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .clip_models import CLIPModel
from utils.dream_degradations import (
    apply_named_degradation,
    clamp01,
    denormalize,
    gaussian_blur_tensor,
    make_eval_degradation,
    make_train_degradation_views,
)


class LowRankExpertPromptBank(nn.Module):
    def __init__(self, num_experts, n_ctx, prompt_dim, rank, init_scale=1e-3):
        super().__init__()
        self.A = nn.Parameter(torch.randn(num_experts, n_ctx, rank) * 0.02)
        self.B = nn.Parameter(torch.randn(num_experts, rank, prompt_dim) * 0.02)
        self.scale = nn.Parameter(torch.full((num_experts, 1, 1), float(init_scale)))

    def residual_matrix(self):
        return torch.matmul(self.A, self.B) * self.scale

    def diversity_loss(self):
        residual = self.residual_matrix().float()
        if residual.shape[0] < 2:
            return residual.sum() * 0.0
        flat = residual.reshape(residual.shape[0], -1)
        flat = F.normalize(flat, dim=1, eps=1e-8)
        cosine = flat @ flat.t()
        mask = ~torch.eye(cosine.shape[0], dtype=torch.bool, device=cosine.device)
        return cosine[mask].pow(2).mean()

    def offdiag_cosine_mean(self):
        residual = self.residual_matrix().float()
        if residual.shape[0] < 2:
            return residual.sum() * 0.0
        flat = F.normalize(residual.reshape(residual.shape[0], -1), dim=1, eps=1e-8)
        cosine = flat @ flat.t()
        mask = ~torch.eye(cosine.shape[0], dtype=torch.bool, device=cosine.device)
        return cosine[mask].abs().mean()

    def forward(self, batch_size, dtype, device):
        residual = self.residual_matrix().to(dtype=dtype, device=device)
        residual = residual.unsqueeze(0).expand(batch_size, -1, -1, -1)
        return residual


class ReliabilityRouter(nn.Module):
    def __init__(self, in_dim, hidden, num_experts, apply_init_bias):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.q_head = nn.Linear(hidden, num_experts)
        self.apply_head = nn.Linear(hidden, 1)
        nn.init.constant_(self.apply_head.bias, apply_init_bias)

    def forward(self, r):
        h = self.net(r)
        q = F.softmax(self.q_head(h), dim=-1)
        apply_logit = self.apply_head(h).squeeze(-1)
        apply_prob = torch.sigmoid(apply_logit)
        return q, apply_prob, apply_logit


class DREAMCSModel(CLIPModel):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.current_epoch = 0
        self.num_experts = args.dream_num_experts
        assert self.num_experts > 0
        assert len(args.dream_expert_names) >= self.num_experts

        prompt_dim = args.vision_width
        n_ctx = args.n_ctx
        feature_dim = self.fc_binary.in_features
        self.dream_feature_dim = feature_dim

        self.dream_expert_bank = LowRankExpertPromptBank(
            num_experts=self.num_experts,
            n_ctx=n_ctx,
            prompt_dim=prompt_dim,
            rank=args.dream_rank,
            init_scale=args.dream_residual_scale_init,
        )
        router_in_dim = 1 + self.num_experts * 4 + 4 + 4
        self.dream_router = ReliabilityRouter(
            in_dim=router_in_dim,
            hidden=args.dream_router_hidden,
            num_experts=self.num_experts,
            apply_init_bias=args.dream_apply_init_bias,
        )

        self.criterion_weight_dict = {
            'loss_dream_clean': args.loss_dream_clean,
            'loss_dream_anchor': args.loss_dream_anchor,
            'loss_dream_expert': args.loss_dream_expert,
            'loss_dream_specialize': args.loss_dream_specialize,
            'loss_dream_anchor_rob': args.loss_dream_anchor_rob,
            'loss_dream_rob': args.loss_dream_rob,
            'loss_dream_inv': args.loss_dream_inv,
            'loss_dream_route': args.loss_dream_route,
            'loss_dream_apply': args.loss_dream_apply,
            'loss_dream_clean_safe': args.loss_dream_clean_safe,
            'loss_dream_res': args.loss_dream_res,
            'loss_dream_div': args.loss_dream_div,
        }
        if args.condition:
            self.criterion_weight_dict['loss_condition'] = args.loss_condition
        if args.use_contrast:
            self.criterion_weight_dict['loss_contrast'] = args.loss_contrast

        if args.dream_anchor_ckpt:
            print('WARNING: dream_anchor_ckpt is plugin/warmstart ablation, not DREAM-CS Standalone.')
            self._load_anchor_checkpoint(args.dream_anchor_ckpt)
        if args.dream_freeze_anchor:
            self._freeze_anchor_parameters()
        self._apply_ablation_trainability()
        if args.tta:
            self.freeze_tta()

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def _load_anchor_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {
                key.replace('module.', '', 1) if key.startswith('module.') else key: value
                for key, value in state_dict.items()
            }
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print('DREAM-CS anchor ablation checkpoint loaded from {}'.format(ckpt_path))
        print('DREAM-CS missing keys:', missing)
        print('DREAM-CS unexpected keys:', unexpected)

    def _freeze_anchor_parameters(self):
        for name, param in self.named_parameters():
            param.requires_grad = name.startswith('dream_')
        print('-----------DREAM-CS freeze anchor ablation mode-----------')
        print([name for name, param in self.named_parameters() if param.requires_grad])

    def _apply_ablation_trainability(self):
        if self.args.dream_disable_router or self.args.dream_disable_expert_correction:
            for param in self.dream_router.parameters():
                param.requires_grad = False
        if self.args.dream_disable_expert_correction:
            for param in self.dream_expert_bank.parameters():
                param.requires_grad = False

    def freeze_tta(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
        for name, param in self.prompt_learner.named_parameters():
            if 'ctx' in name:
                param.requires_grad = True
        trainable = [name for name, param in self.named_parameters() if param.requires_grad]
        print('-----------freezen DREAM-CS TTA mode-----------')
        print(trainable)

    def _make_anchor_ctx(self, image):
        image_vp = image.type(self.dtype)
        shared_ctx, deep_prompts = self.prompt_learner()

        pred_bias = None
        cond_bias = None
        if self.conditional_ctx is not None:
            cond_bias, pred_bias = self.conditional_ctx(image_vp)

        if shared_ctx is not None:
            shared_ctx = shared_ctx.expand(image_vp.shape[0], -1, -1)

        if cond_bias is not None:
            anchor_ctx = shared_ctx + cond_bias if shared_ctx is not None else cond_bias
        else:
            anchor_ctx = shared_ctx

        assert anchor_ctx is not None, 'DREAM-CS Standalone requires shallow visual prompt context.'
        return image_vp, anchor_ctx, deep_prompts, pred_bias

    def _encode_with_context(self, image_vp, ctx, deep_prompts):
        chunk = int(getattr(self.args, 'dream_expert_forward_chunk', 0))
        if chunk > 0 and image_vp.shape[0] > chunk:
            feats = []
            for start in range(0, image_vp.shape[0], chunk):
                end = start + chunk
                feat, _ = self.image_encoder(image_vp[start:end], ctx[start:end], deep_prompts)
                feats.append(feat)
            return torch.cat(feats, dim=0)

        feat, _ = self.image_encoder(image_vp, ctx, deep_prompts)
        return feat

    def _image_reliability_stats(self, image):
        with torch.no_grad():
            img = clamp01(denormalize(image.detach(), self.args.dataset)).float()
            gray = img[:, 0:1] * 0.2989 + img[:, 1:2] * 0.5870 + img[:, 2:3] * 0.1140
            blur_gray = gaussian_blur_tensor(gray, kernel_size=3).float()
            blur_img = gaussian_blur_tensor(img, kernel_size=3).float()

            highfreq = (gray - blur_gray).abs().flatten(1).mean(dim=1, keepdim=True)
            blur_residual = (img - blur_img).abs().flatten(1).mean(dim=1, keepdim=True)
            contrast = gray.flatten(1).std(dim=1, unbiased=False).unsqueeze(1)

            bsz, _, height, width = gray.shape
            boundary_terms = []
            if width > 8:
                cols = torch.arange(8, width, 8, device=gray.device)
                boundary_terms.append((gray[:, :, :, cols] - gray[:, :, :, cols - 1]).abs().mean(dim=(1, 2, 3)))
            if height > 8:
                rows = torch.arange(8, height, 8, device=gray.device)
                boundary_terms.append((gray[:, :, rows, :] - gray[:, :, rows - 1, :]).abs().mean(dim=(1, 2, 3)))
            if len(boundary_terms) == 0:
                boundary_diff = gray.new_zeros(bsz)
            else:
                boundary_diff = torch.stack(boundary_terms, dim=0).mean(dim=0)

            non_boundary_v = (gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs().mean(dim=(1, 2, 3))
            non_boundary_h = (gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs().mean(dim=(1, 2, 3))
            non_boundary = 0.5 * (non_boundary_v + non_boundary_h)
            blockiness = (boundary_diff - non_boundary).clamp_min(0.0).unsqueeze(1)

            return torch.cat([highfreq, blur_residual, contrast, blockiness], dim=1).detach()

    def _image_counterfactual_stats(self, image):
        with torch.no_grad():
            clean_stats = self._image_reliability_stats(image)
            jpeg75 = apply_named_degradation(image, 'jpeg75', self.args.dataset)
            resize = apply_named_degradation(image, 'resize', self.args.dataset)
            jpeg75_stats = self._image_reliability_stats(jpeg75)
            resize_stats = self._image_reliability_stats(resize)
            highfreq_decay = clean_stats[:, 0:1] - jpeg75_stats[:, 0:1]
            contrast_decay = clean_stats[:, 2:3] - resize_stats[:, 2:3]
            blockiness_jpeg75 = jpeg75_stats[:, 3:4]
            blur_residual_decay = clean_stats[:, 1:2] - jpeg75_stats[:, 1:2]
            return torch.cat([highfreq_decay, contrast_decay, blockiness_jpeg75, blur_residual_decay], dim=1).detach()

    def _build_reliability_features(self, image, z0, ze, h0, he):
        with torch.no_grad():
            delta = ze - z0.unsqueeze(1)
            anchor_margin = (z0.sigmoid() - 0.5).abs().mul(2.0).unsqueeze(1)
            expert_margins = (ze.sigmoid() - 0.5).abs().mul(2.0)
            abs_delta = delta.abs()
            signed_delta = delta
            feature_residual_norm = (he.float() - h0.unsqueeze(1).float()).pow(2).mean(dim=-1).sqrt()
            feature_residual_norm = feature_residual_norm / math.sqrt(float(self.dream_feature_dim))
            image_stats = self._image_reliability_stats(image).to(device=z0.device, dtype=z0.dtype)
            counterfactual_stats = self._image_counterfactual_stats(image).to(device=z0.device, dtype=z0.dtype)
            rel = torch.cat([
                anchor_margin,
                expert_margins,
                abs_delta,
                signed_delta,
                feature_residual_norm.to(dtype=z0.dtype),
                image_stats,
                counterfactual_stats,
            ], dim=1)
        return rel.detach(), image_stats.detach(), counterfactual_stats.detach(), feature_residual_norm.detach()

    def forward_once(self, image, force_anchor=False):
        bsz = image.shape[0]
        k = self.num_experts
        image_vp, anchor_ctx, deep_prompts, pred_bias = self._make_anchor_ctx(image)

        residual = self.dream_expert_bank(bsz, dtype=anchor_ctx.dtype, device=anchor_ctx.device)
        expert_ctx = anchor_ctx.unsqueeze(1) + residual
        all_ctx = torch.cat([anchor_ctx.unsqueeze(1), expert_ctx], dim=1)

        prompt_len = anchor_ctx.shape[1]
        prompt_dim = anchor_ctx.shape[2]
        assert anchor_ctx.shape == (bsz, prompt_len, prompt_dim)
        assert residual.shape == (bsz, k, prompt_len, prompt_dim)
        assert expert_ctx.shape == (bsz, k, prompt_len, prompt_dim)

        all_ctx = all_ctx.reshape(bsz * (k + 1), prompt_len, prompt_dim)
        all_img = image_vp.unsqueeze(1).expand(-1, k + 1, -1, -1, -1)
        all_img = all_img.reshape(bsz * (k + 1), *image_vp.shape[1:])
        assert all_ctx.shape == (bsz * (k + 1), prompt_len, prompt_dim)
        assert all_img.shape[0] == bsz * (k + 1)

        all_feat = self._encode_with_context(all_img, all_ctx, deep_prompts)
        feat_dim = all_feat.shape[-1]
        assert all_feat.shape == (bsz * (k + 1), feat_dim)
        all_feat = all_feat.reshape(bsz, k + 1, feat_dim)
        h0 = all_feat[:, 0, :]
        he = all_feat[:, 1:, :]
        assert h0.shape == (bsz, feat_dim)
        assert he.shape == (bsz, k, feat_dim)

        z0 = self.fc_binary(h0).squeeze(-1)
        ze = self.fc_binary(he.reshape(bsz * k, feat_dim)).reshape(bsz, k)
        assert z0.shape == (bsz,)
        assert ze.shape == (bsz, k)

        rel = None
        image_stats = z0.new_zeros(bsz, 4)
        counterfactual_stats = z0.new_zeros(bsz, 4)
        feature_residual_norm = (he.float() - h0.unsqueeze(1).float()).pow(2).mean(dim=-1).sqrt()
        feature_residual_norm = feature_residual_norm / math.sqrt(float(self.dream_feature_dim))
        delta_raw = ze - z0.unsqueeze(1)
        delta = torch.clamp(
            delta_raw,
            min=-self.args.dream_delta_clip,
            max=self.args.dream_delta_clip,
        )

        if self.args.dream_disable_expert_correction or force_anchor:
            q = z0.new_full((bsz, k), 1.0 / float(k))
            apply_raw = z0.new_zeros(bsz)
            apply_eff = z0.new_zeros(bsz)
            apply_logit = z0.new_full((bsz,), -20.0)
            z = z0
        else:
            rel, image_stats, counterfactual_stats, feature_residual_norm = self._build_reliability_features(image, z0, ze, h0, he)
            q, apply_raw, apply_logit = self.dream_router(rel)
            if self.args.dream_disable_router:
                q = z0.new_full((bsz, k), 1.0 / float(k))
                apply_raw = z0.new_ones(bsz)
                apply_logit = z0.new_full((bsz,), 20.0)

            apply_eff = apply_raw
            if (
                self.training
                and self.args.dream_use_apply_floor
                and self.current_epoch < self.args.dream_warmup_epochs
            ):
                floor = float(self.args.dream_apply_train_floor)
                apply_eff = apply_raw + floor * (1.0 - apply_raw)

            z = z0 + apply_eff * (q * delta).sum(dim=1)

        residual_f = residual.float()
        delta_clip = (delta_raw.abs() > float(self.args.dream_delta_clip)).float()
        assert q.shape == (bsz, k)
        assert apply_eff.shape == (bsz,)
        assert z.shape == (bsz,)
        return {
            'logits': z.unsqueeze(-1),
            'logits_flat': z,
            'anchor_logits': z0,
            'expert_logits': ze,
            'q': q,
            'apply': apply_eff,
            'apply_eff': apply_eff,
            'apply_raw': apply_raw,
            'apply_logit': apply_logit,
            'delta_raw': delta_raw,
            'delta_clamped': delta,
            'delta_abs_mean': delta_raw.abs().mean(),
            'delta_clip_frac': delta_clip.mean(),
            'delta_clip_frac_per_expert': delta_clip.mean(dim=0),
            'h0': h0,
            'he': he,
            'prompt_residual_norm': residual_f.pow(2).mean(),
            'residual_norm_per_expert': residual_f.pow(2).mean(dim=(0, 2, 3)),
            'residual_scale_per_expert': self.dream_expert_bank.scale.detach().view(-1).to(device=z0.device, dtype=z0.dtype),
            'residual_cos_offdiag': self.dream_expert_bank.offdiag_cosine_mean().to(device=z0.device, dtype=z0.dtype),
            'feature_residual_norm_per_expert': feature_residual_norm.to(device=z0.device, dtype=z0.dtype),
            'image_stats': image_stats,
            'counterfactual_stats': counterfactual_stats,
            'rel_features': rel,
            'pred_bias': pred_bias,
        }

    def forward(self, image):
        if not self.training and getattr(self.args, 'dream_eval_degradation', 'none') != 'none':
            image = make_eval_degradation(image, self.args)

        clean_out = self.forward_once(image)
        if (
            self.training
            and (not getattr(self.args, 'tta', False))
            and (not self.args.dream_disable_robust)
        ):
            deg_items = make_train_degradation_views(image, self.args, return_names=True)
            deg_outputs = []
            for item in deg_items:
                if isinstance(item, dict):
                    deg_out = self.forward_once(item['image'])
                    deg_out['deg_name'] = item['name']
                else:
                    deg_out = self.forward_once(item)
                    deg_out['deg_name'] = 'unknown'
                deg_outputs.append(deg_out)
            clean_out['deg_outputs'] = deg_outputs
        return clean_out

    def _smooth_targets(self, targets):
        targets = targets.float()
        if getattr(self.args, 'smooth', False):
            targets = targets * 0.9 + 0.05
        return targets

    def _bce_loss(self, logits, targets, reduction='mean'):
        targets = self._smooth_targets(targets).to(device=logits.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, targets, reduction=reduction)

    def _zero(self, outputs):
        return outputs['logits_flat'].sum() * 0.0

    def _expert_specialization_loss(self, outputs, y, ce_expert_clean):
        zero = self._zero(outputs)
        if self.args.dream_disable_specialization:
            return zero
        k = self.num_experts
        deg_outputs = outputs.get('deg_outputs', [])
        if k < 3:
            return ce_expert_clean.mean()

        light = ['jpeg90', 'jpeg75']
        strong = ['jpeg50', 'resize', 'blur', 'quant', 'webp']
        fine_losses = [self._bce_loss(outputs['expert_logits'][:, 0], y)]
        stable_losses = []
        consensus_logits = [outputs['expert_logits'][:, 2]]

        for deg_out in deg_outputs:
            name = deg_out.get('deg_name', 'unknown')
            if name in light:
                fine_losses.append(self._bce_loss(deg_out['expert_logits'][:, 0], y))
            if name in strong:
                stable_losses.append(self._bce_loss(deg_out['expert_logits'][:, 1], y, reduction='none'))
            consensus_logits.append(deg_out['expert_logits'][:, 2])

        loss_fine = torch.stack(fine_losses).mean()
        if len(stable_losses) == 0:
            if len(deg_outputs) > 0:
                stable_losses = [
                    self._bce_loss(deg_out['expert_logits'][:, 1], y, reduction='none')
                    for deg_out in deg_outputs
                ]
            else:
                stable_losses = [ce_expert_clean[:, 1]]
        loss_stable = torch.stack(stable_losses, dim=0).max(dim=0)[0].mean()

        consensus_stack = torch.stack(consensus_logits, dim=0)
        consensus_ce = torch.stack([
            self._bce_loss(logit, y, reduction='none') for logit in consensus_logits
        ], dim=0).mean()
        consensus_var = consensus_stack.var(dim=0, unbiased=False).mean()
        loss_consensus = consensus_ce + 0.1 * consensus_var

        return (loss_fine + loss_stable + loss_consensus) / 3.0

    def get_criterion(self, outputs, targets):
        loss_dic = {}
        logits = outputs['logits_flat']
        zero = logits.sum() * 0.0
        y = targets.float().to(device=logits.device).view(-1)
        k = self.num_experts
        y_expand = y[:, None].expand(-1, k)

        ce_final_clean = self._bce_loss(outputs['logits_flat'], y, reduction='none')
        ce_anchor_clean = self._bce_loss(outputs['anchor_logits'], y, reduction='none')
        ce_expert_clean = self._bce_loss(outputs['expert_logits'], y_expand, reduction='none')

        loss_dic['loss_dream_clean'] = ce_final_clean.mean()
        loss_dic['loss_dream_anchor'] = ce_anchor_clean.mean()
        loss_dic['loss_dream_expert'] = zero
        loss_dic['loss_dream_specialize'] = zero
        loss_dic['loss_dream_anchor_rob'] = zero

        if 'loss_condition' in self.criterion_weight_dict:
            pred_bias = outputs.get('pred_bias', None)
            loss_dic['loss_condition'] = self._bce_loss(pred_bias.view(-1), y) if pred_bias is not None else zero

        if 'loss_contrast' in self.criterion_weight_dict:
            contrast = self.contrastive_loss(outputs['h0'], targets)
            if not torch.is_tensor(contrast):
                contrast = zero + float(contrast)
            loss_dic['loss_contrast'] = contrast

        deg_outputs = outputs.get('deg_outputs', [])
        deg_expert_ce = []
        if len(deg_outputs) > 0:
            ce_views = [ce_final_clean]
            anchor_ce_views = [ce_anchor_clean]
            expert_ce_views = [ce_expert_clean]
            for deg_out in deg_outputs:
                ce_views.append(self._bce_loss(deg_out['logits_flat'], y, reduction='none'))
                anchor_ce_views.append(self._bce_loss(deg_out['anchor_logits'], y, reduction='none'))
                expert_ce = self._bce_loss(deg_out['expert_logits'], y_expand, reduction='none')
                expert_ce_views.append(expert_ce)
                deg_expert_ce.append(expert_ce.mean())

            loss_dic['loss_dream_rob'] = torch.stack(ce_views, dim=0).max(dim=0)[0].mean()
            loss_dic['loss_dream_anchor_rob'] = torch.stack(anchor_ce_views, dim=0).max(dim=0)[0].mean()
            logits_stack = torch.stack(
                [outputs['logits_flat']] + [deg_out['logits_flat'] for deg_out in deg_outputs],
                dim=0,
            )
            loss_dic['loss_dream_inv'] = logits_stack.var(dim=0, unbiased=False).mean()
        else:
            ce_views = [ce_final_clean]
            anchor_ce_views = [ce_anchor_clean]
            expert_ce_views = [ce_expert_clean]
            loss_dic['loss_dream_rob'] = zero
            loss_dic['loss_dream_inv'] = zero

        if not self.args.dream_disable_expert_aux:
            clean_expert_loss = ce_expert_clean.mean()
            if len(deg_expert_ce) > 0:
                loss_dic['loss_dream_expert'] = 0.7 * clean_expert_loss + 0.3 * torch.stack(deg_expert_ce).mean()
            else:
                loss_dic['loss_dream_expert'] = clean_expert_loss
        loss_dic['loss_dream_specialize'] = self._expert_specialization_loss(outputs, y, ce_expert_clean)

        if self.args.dream_disable_clean_safe:
            loss_dic['loss_dream_clean_safe'] = zero
        else:
            anchor_ref = ce_anchor_clean.detach()
            clean_safe_violation = ce_final_clean > (anchor_ref + self.args.dream_clean_safe_margin)
            clean_safe = F.relu(ce_final_clean - anchor_ref + self.args.dream_clean_safe_margin).mean()
            loss_dic['loss_dream_clean_safe'] = clean_safe

        route_disabled = (
            self.args.dream_disable_route_loss
            or self.args.dream_disable_router
            or self.args.dream_disable_expert_correction
            or self.current_epoch < self.args.dream_router_start_epoch
        )

        i_clean = ce_anchor_clean.detach()[:, None] - ce_expert_clean.detach()
        anchor_worst = torch.stack(anchor_ce_views, dim=0).max(dim=0)[0]
        expert_worst = torch.stack(expert_ce_views, dim=0).max(dim=0)[0]
        i_rob = anchor_worst.detach()[:, None] - expert_worst.detach()
        improvement = torch.minimum(i_clean, i_rob)
        active = improvement.max(dim=1).values > self.args.dream_route_margin
        tau = max(float(self.args.dream_route_tau), 1e-6)
        target_q = F.softmax(F.relu(improvement / tau), dim=1)

        if route_disabled:
            loss_dic['loss_dream_route'] = zero
            loss_dic['loss_dream_apply'] = zero
        else:
            if active.any():
                q_active = outputs['q'][active].clamp_min(1e-8)
                loss_dic['loss_dream_route'] = F.kl_div(
                    torch.log(q_active),
                    target_q[active],
                    reduction='batchmean',
                )
            else:
                loss_dic['loss_dream_route'] = zero
            loss_dic['loss_dream_apply'] = F.binary_cross_entropy_with_logits(
                outputs['apply_logit'],
                active.float().to(device=outputs['apply_logit'].device, dtype=outputs['apply_logit'].dtype),
            )

        loss_dic['loss_dream_res'] = outputs['prompt_residual_norm']
        loss_dic['loss_dream_div'] = self.dream_expert_bank.diversity_loss().to(device=logits.device, dtype=logits.dtype)

        loss_dic['stat_active_rate'] = active.float().mean()
        if self.args.dream_disable_clean_safe:
            loss_dic['stat_clean_safe_violation'] = zero
        else:
            loss_dic['stat_clean_safe_violation'] = clean_safe_violation.float().mean()
        loss_dic['stat_anchor_ce'] = ce_anchor_clean.mean()
        loss_dic['stat_final_ce'] = ce_final_clean.mean()
        for idx in range(min(3, k)):
            loss_dic['stat_i_clean_e{}'.format(idx)] = i_clean[:, idx].mean()
            loss_dic['stat_i_rob_e{}'.format(idx)] = i_rob[:, idx].mean()
            loss_dic['stat_target_q_e{}'.format(idx)] = target_q[:, idx].mean()
            loss_dic['stat_expert_ce_e{}'.format(idx)] = ce_expert_clean[:, idx].mean()

        return loss_dic
