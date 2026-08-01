# EMA and effective-batch contract

The effective batch is always recorded as `per_device_batch_size ×
gradient_accumulation_steps × world_size`. A configured target that differs from
the runtime calculation fails before training.

EMA is optional and updates only after an optimizer update. Its state is stored in
the optional `ema_state_dict` checkpoint field, leaving `model_state_dict` as the
raw weights for legacy readers. Raw evaluation is the default; callers can use the
EMA context manager to evaluate averaged weights without overwriting the raw model.
Older checkpoints without EMA state continue loading with EMA initialized from the
loaded raw model.
