"""Notebook runner for patchcore using jupyter-ml infrastructure."""

from pathlib import Path

from jupyter_ml.runner import capture_outputs
from jupyter_ml.notebook import notebook_hash, notebook_diff


NOTEBOOKS = [
    "mlx_anomalib_notebook.ipynb",
]


def run_notebooks() -> dict:
    """Run all patchcore notebooks and capture outputs."""
    results = {}
    for notebook in NOTEBOOKS:
        path = Path(__file__).parent.parent / notebook
        if path.exists():
            outputs = capture_outputs(str(path))
            results[notebook] = {
                "outputs": outputs,
                "cells": len(outputs),
            }
    return results


def check_notebook_changes(notebook: str, baseline_hash: str) -> bool:
    """Check if notebook has changed from baseline."""
    path = Path(__file__).parent.parent / notebook
    current_hash = notebook_hash(str(path))
    return current_hash != baseline_hash


def main() -> None:
    """Main entry point."""
    print("=== Patchcore Notebook Runner (using jupyter-ml) ===\n")

    # Run notebooks
    print("Running notebooks...")
    results = run_notebooks()

    for notebook, data in results.items():
        print(f"  {notebook}: {data['cells']} cells with outputs")

    print("\nDone!")


if __name__ == "__main__":
    main()
