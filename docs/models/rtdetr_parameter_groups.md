# RT-DETRv2 parameter-group discovery

PR 16 is diagnostic only. Trainable parameters are enumerated once by identity,
assigned to `backbone` when their qualified name contains the documented backbone
boundary, and otherwise assigned to `detector`. Empty names, unknown classifier
results, duplicates, and count mismatches fail before optimizer construction.

The report records sorted names and both tensor and scalar-parameter counts. It
sets `differential_lr_enabled` to false: the existing global learning rate remains
the only active optimizer policy until PR 17.
