# Best checkpoint and early-stopping contract

`last.pth` is always the resumable state. `best_raw.pth` preserves the highest raw
validation mAP, while `best_ema.pth` is optional and can only be selected from EMA
metrics. `best.pt` and `best_map.pth` remain compatibility copies of `best_raw.pth`.

Checkpoint selection and early stopping use separate persisted states. Early
stopping compares raw validation mAP using the configured patience and minimum
delta; resuming with a different policy is rejected. Metric schema v2 records the
raw/EMA weight variant to prevent comparison tables from silently mixing them.
