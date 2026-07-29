# Contributing

1. Create a focused branch and add or update tests.
2. Run `python scripts/clean_notebooks.py notebooks`, then `make verify`.
3. Never commit datasets, checkpoints, credentials, Drive tokens, or unverified benchmark numbers.
4. Document the exact source, license, checksum, preprocessing, hardware, and evaluation track for new models or results.
5. Keep framework-specific code behind adapters and reusable code outside notebooks.
6. Treat `.ipynb` files as canonical and never commit executed notebook outputs.
