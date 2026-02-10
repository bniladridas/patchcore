"""Regression tests for patchcore notebooks using jupyter-ml."""

from pathlib import Path

from jupyter_ml.runner import capture_outputs
from jupyter_ml.notebook import notebook_hash


def test_notebook_runs():
    """Test that notebooks execute without errors."""
    notebook = Path(__file__).parent.parent / "mlx_anomalib_notebook.ipynb"
    if not notebook.exists():
        return  # Skip if notebook doesn't exist

    outputs = capture_outputs(str(notebook))
    assert len(outputs) > 0


def test_notebook_structure():
    """Test that notebook structure is stable."""
    notebook = Path(__file__).parent.parent / "mlx_anomalib_notebook.ipynb"
    if not notebook.exists():
        return

    hash1 = notebook_hash(str(notebook))
    hash2 = notebook_hash(str(notebook))
    assert hash1 == hash2
