# RT-DETRv2 optimizer recipe v2

Recipe v2 uses AdamW with a detector learning rate and a reviewed 0.1 backbone
multiplier. Warm-up and cosine decay are expressed in optimizer updates against a
single full-run horizon, so a short search trial is an exact prefix of the final
schedule. The scheduler advances only after an optimizer update.

The checked-in performance and smoke policies set a 0.1 gradient-norm clip. The
legacy global-LR recipe remains the default unless `recipe_version` is explicitly
`rtdetr_recipe_v2`; reverting the HPO/final override restores it without changing
checkpoint files.
