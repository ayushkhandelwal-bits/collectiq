# Deployment

## Streamlit Community Cloud (free)
1. Push the repository to GitHub (public).
2. At https://share.streamlit.io choose **New app** → repository
   `ayushkhandelwal-bits/collectiq`, branch `main`, main file `app.py`.
3. Under **Advanced settings** choose Python 3.13 (the version the project was built and tested on; 3.11+ is required by pandas 3). Optionally add the
   environment variable `COLLECTIQ_PROFILE = "demo"` (the default for cloud bootstrap).
4. Deploy. On first load the app shows **Build data and models**. One click runs the
   pipeline for the demo profile (20,000 loans): generation, validation, database load,
   training, scoring and a default optimisation. This takes roughly one to two minutes;
   afterwards pages load from cached outputs.
5. Verify the URL opens, then visit every page.

No secrets, external databases or paid services are needed. Nothing large is committed:
the database, raw parquet and model artifacts are rebuilt from code.

### Resource notes
* The demo profile keeps memory well under 1 GB.
* The cloud filesystem is ephemeral; a restart requires the one-click rebuild again and
  clears saved optimisation runs.

## Local
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.pipeline.run_pipeline          # full profile, ~1 minute
streamlit run app.py
```
Use `COLLECTIQ_PROFILE=demo` for a lighter build, or `--skip-train` to reuse registered
models.

## Verification checklist
* `pytest` passes.
* `ruff check .` passes.
* App starts and all eight pages render; Run optimization and Run scenario complete.
* Data & Governance shows "Dictionary verified against the live database".
