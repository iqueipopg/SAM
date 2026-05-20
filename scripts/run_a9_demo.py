#!/usr/bin/env python3
"""
Standalone runner for Agent9Explainer using the test fixture.
Generates papers_raw.xlsx + explainability.md from
tests/conftest.py::full_pipeline_data into outputs/eval/medio/A9_demo/.
"""
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

# Import fixture from conftest by manually invoking it. pytest fixtures
# are normally invoked by pytest; here we call the generator directly
# if it's plain data, or use a fallback if it's a parametrized fixture.
def _load_pipeline_data():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_conftest_a9",
        str(PROJECT_ROOT / "tests" / "conftest.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # The fixture function is usually def full_pipeline_data():
    fn = getattr(mod, "full_pipeline_data", None)
    if fn is None:
        raise SystemExit("full_pipeline_data fixture not found in conftest.py")
    # If it's a pytest fixture decorated, .__wrapped__ has the raw fn
    raw = getattr(fn, "__wrapped__", fn)
    return raw()

def main():
    pipeline_data = _load_pipeline_data()
    out_dir = PROJECT_ROOT / "outputs" / "eval" / "medio" / "A9_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    from agents.agent9_explainer import Agent9Explainer
    agent = Agent9Explainer()
    result = agent.run(
        pipeline_data=pipeline_data,
        run_id="A9_demo_blockE_fixture",
        output_dir=str(out_dir),
    )

    print("=== A9 demo run completed ===")
    print(f"Excel:  {result.get('excel_path')}")
    print(f"Report: {result.get('report_path')}")
    print(f"Stats:  {result.get('stats')}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
