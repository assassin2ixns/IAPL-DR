# DREAM-CS Standalone Quick Diagnosis

## Best / Last Final vs Anchor
best_epoch=1 best_final_acc=0.9062865014476453 best_final_ap=0.9571090263999824 anchor_acc=0.9062865014476453 anchor_ap=0.9571090263999824
last_epoch=1 final_acc=0.8985625 final_ap=0.9817227760744136 racc=0.9982500000000001 facc=0.798875 rf_gap=0.19937500000000002 ece=0.0709412115777377 brier=0.0793443188898056

## No-Regret
help=0.0 harm=0.0 CAVR=0.0 no_regret=0.0 high_conf_anchor_harm=0.0

## Router
apply=2.5952622218312625e-06 q_entropy=0.9421209287906345 q=[0.5996542266979814, 0.24118703526444735, 0.1591587378587574] q_top1=[1.0, 0.0, 0.0]

## Experts
expert_oracle_gain=3.90443483564799e-05 final_oracle_gap=0.00010975419743076319 expert_better_rates=[0.3900625, 0.404, 0.3999375]

## Red Flags
- epoch=0 deg=none test_domain_checkpoint_selection: WARNING: checkpoint_best_ap uses evaluation domains and should not be used for final paper selection.
- epoch=0 deg=none test_domain_checkpoint_selection: WARNING: checkpoint_best_acc uses evaluation domains and should not be used for final paper selection.
- epoch=1 deg=none test_domain_checkpoint_selection: WARNING: checkpoint_best_ap uses evaluation domains and should not be used for final paper selection.
- epoch=1 deg=none test_domain_checkpoint_selection: WARNING: checkpoint_best_acc uses evaluation domains and should not be used for final paper selection.

## Recommendation
- Expert dead: raise loss_dream_expert or delay dream_router_start_epoch.
