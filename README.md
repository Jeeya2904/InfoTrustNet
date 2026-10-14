# InfoTrustNet

**A multi-hybrid misinformation-detection demo combining a text classifier, a knowledge graph, reinforcement-learning-based signal fusion, and explainable AI.**

> **Scope note:** The bundled dataset is synthetic and intended for a reproducible software demo. Its metrics are not evidence of real-world misinformation-detection performance. Verify claims with trustworthy sources before making real decisions.

## Features

- TF-IDF text classifier (offline-friendly default)
- Knowledge-graph evidence for supported or contradicted facts
- Reinforcement-learning policy for combining model, graph, and source signals
- Human-readable explanations and token-occlusion estimates
- CLI for training, evaluation, prediction, and explanations
- Optional Streamlit web interface
- Pytest tests and GitHub Actions CI workflow

## Quick start (Windows + VS Code)

1. Install Python 3.11+ and the VS Code Python extension.
2. Open this repository folder in VS Code.
3. Create and activate a virtual environment, then install dependencies:
   ```powershell
   py -3 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Train the demo model and run sample predictions:
   ```powershell
   python main.py train
   python main.py demo
   ```
5. Run tests with `python -m pytest -q`.

If PowerShell blocks activation, use `.venv\Scripts\python.exe main.py train`.

## macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py train
python main.py demo
python -m pytest -q
```

## CLI reference

```bash
python main.py train
python main.py evaluate
python main.py predict --text "A claim to check" --source reuters.com
python main.py explain --text "A claim to check" --source reuters.com
python main.py demo
```

CSV input must contain `text` and `label` columns; `label=1` means misinformation and `label=0` means reliable. An optional `source` column can contain a source domain.

## Optional web UI

```bash
pip install -r requirements-app.txt
python main.py train
streamlit run app.py
```

## Project layout

- `main.py` — CLI entry point
- `app.py` — optional Streamlit UI
- `infotrustnet/` — model, knowledge graph, RL, XAI, data and evaluation modules
- `data/claims.csv` — example dataset
- `tests/` — automated tests
- `artifacts/` — generated model and metrics (not committed)
- `.vscode/` — VS Code launch settings and tasks
- `.github/workflows/ci.yml` — test workflow

## License

MIT
