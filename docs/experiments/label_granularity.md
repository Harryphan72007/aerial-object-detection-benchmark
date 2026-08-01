# Label-granularity ablation

The direct configuration trains and evaluates in the merged person/vehicle class
space. The ablation configuration trains on the ten original VisDrone categories,
then maps pedestrian and people to person and the eight vehicle-like categories to
vehicle for evaluation. Ignored category IDs 0 and 11 remain excluded.

Every metric artifact must carry the label-space manifest. Direct and merged
results have different `ablation_id` and output namespaces, so comparison code
cannot present them as the same experiment.
