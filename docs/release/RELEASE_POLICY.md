# Release policy

Every pull request must pass the `static`, `cpu-tests`, `notebooks`, and `security`
workflows. Repository administrators should configure the four corresponding job names
as required branch-protection checks:

- `static / static-and-schema`
- `cpu-tests / cpu-tests`
- `notebooks / notebook-contracts`
- `security / security-and-artifacts`

These checks validate configuration and JSON documents, CPU behavior, notebook syntax
and smoke execution, legacy inventory, prohibited artifacts, secrets, and result bundles.
They require no GPU, dataset, Google Drive, or private credentials.

GPU validation follows [GPU_VALIDATION_CHECKLIST.md](GPU_VALIDATION_CHECKLIST.md) and is
performed manually in Colab. Releases that affect model execution, checkpoint/resume,
or evaluator numerics require linked GPU evidence or an explicit release-manager waiver.

Release artifacts are immutable, versioned, lightweight result bundles. Raw datasets,
checkpoints, predictions, caches, logs, credentials, and workstation paths are prohibited.
Rollback is a normal revert of the release commit; generated external artifacts remain
isolated by versioned namespaces and may be retired separately.
