# Patchcore Infrastructure

This project uses [jupyter-ml](https://github.com/bniladridas/jupyter-ml) for notebook infrastructure.

## What jupyter-ml Provides

- **Notebook execution** — Run notebooks programmatically
- **Output capture** — Track metrics across runs
- **Change detection** — Detect notebook drift
- **Regression testing** — Validate notebooks in CI

## Usage

### Run notebooks and capture outputs

```python
from jupyter_ml.runner import capture_outputs
from jupyter_ml.notebook import notebook_hash

outputs = capture_outputs("mlx_anomalib_notebook.ipynb")
current_hash = notebook_hash("mlx_anomalib_notebook.ipynb")
```

### CLI

```bash
python scripts/notebook_runner.py
```

## Tests

```bash
pytest tests/test_notebooks.py
```

## Integration

jupyter-ml is integrated via `requirements.txt`:

```
jupyter-ml @ file:///path/to/jupyter-ml
```

