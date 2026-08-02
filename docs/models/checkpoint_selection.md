# Best checkpoint and early-stopping contract

New runs use one rolling `last.pth` while active. It contains the complete resume
state and the validation-mAP-selected raw model state. After successful validation,
that selected state is materialized as `best.pth`; duplicate and resume checkpoint
files are then removed. APtiny remains a metric and does not select another file.

Completed manifests use `checkpoint_best`. Legacy `best_map.pth`, `best_raw.pth`,
and explicit aliases remain readable through the compatibility resolver but are not
written by new training workflows.

Checkpoint selection and early stopping use separate persisted states. Early
stopping compares raw validation mAP using the configured patience and minimum
delta; resuming with a different policy is rejected. Metric schema v2 records the
raw/EMA weight variant to prevent comparison tables from silently mixing them. The
controlled benchmark currently declares `weight_variant: raw`.
