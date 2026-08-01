# Security policy

## Reporting a vulnerability

Do not open a public issue for a vulnerability, leaked credential, private
dataset path, or exposed research participant information. Use GitHub's private
security-advisory workflow for this repository. Include the affected revision,
reproduction steps, impact, and the smallest safe remediation you know.

If a credential has already been exposed, revoke and rotate it immediately.
Removing it from the latest commit is not sufficient because Git history and
forks may retain it. Coordinate any required history remediation with the
repository owner; contributors must not rewrite shared history independently.

## Repository data boundary

GitHub contains source, notebooks, configs, schemas, tests, documentation, and
small reviewed fixtures. Google Drive contains datasets, weights, checkpoints,
predictions, databases, logs, caches, and generated experiment artifacts.

Never commit GitHub tokens, Google credentials, service-account files, private
keys, `.env` files, signed URLs, Drive authentication outputs, or credentials
embedded in repository remotes. Use environment variables or the platform's
secret store and keep diagnostic output out of commits.

Run before every commit:

```bash
python -m scripts.validation.check_prohibited_files
python -m scripts.scan_repository_secrets
```

The checks reduce risk but do not replace review. Report false negatives as
security issues and false positives as ordinary validation bugs.
