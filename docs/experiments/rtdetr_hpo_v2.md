# Persistent RT-DETR Optuna search v2

RT-DETR recipe v2 searches detector LR, backbone multiplier, weight decay,
warm-up steps, and gradient clipping. It uses protocol `rtdetr_optuna_v2`, the
database `study_v2.db`, and a distinct root from the legacy two-stage study.

Each trial launches a fresh training orchestrator with resume disabled. Known CUDA
OOM and numerical-divergence failures are persisted as pruned trials; unexpected
errors remain failed and propagate. After every trial, SQLite's backup API creates
an atomic `snapshots/study_v2_latest.db` copy suitable for Drive persistence.

Study metadata binds the dataset hashes, environment fingerprint, source commit,
search-space hash, recipe hash, objective, and storage policy. Reconnection rejects
scientific-contract drift while retaining a metadata history for non-contract drift.
