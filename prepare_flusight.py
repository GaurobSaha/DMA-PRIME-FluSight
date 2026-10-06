"""Run the submission parts of the notebooks without interactive plotting."""
from pathlib import Path
import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["RF", "XGBoost", "Both"], default="Both")
    parser.add_argument("--mode", choices=["dataset", "live", "retrospective"], default="dataset",
                        help="Dataset forecasts from the latest common data week; Live checks the current submission week.")
    parser.add_argument("--reference-date", help="In Dataset mode this must be one week after the latest common observation.")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--refresh-config", action="store_true")
    parser.add_argument("--trials", type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    os.environ["FLUSIGHT_ROOT"] = str(root)
    os.environ["FLUSIGHT_MODE"] = args.mode
    os.environ["FLUSIGHT_CHECK_ONLY"] = "1" if args.check_only else "0"
    os.environ["MPLBACKEND"] = "Agg"
    import IPython.display
    import matplotlib.pyplot as plt
    IPython.display.display = lambda *items, **kwargs: None
    plt.show = lambda *items, **kwargs: plt.close("all")
    if args.reference_date:
        os.environ["FLUSIGHT_REFERENCE_DATE"] = args.reference_date
    if args.data_dir:
        os.environ["FLUSIGHT_DATA_DIR"] = str(args.data_dir.resolve())
    if args.trials is not None:
        os.environ["FLUSIGHT_TRIALS"] = str(args.trials)
    from flusight_submission import refresh_hub_config
    if args.refresh_config:
        refresh_hub_config(root / "hub-config-cache")
    if args.mode == "live" and not (root / "hub-config-cache" / "tasks.json").is_file():
        raise ValueError("Run with -RefreshHubConfig once to download current official hub configuration.")
    models = ["RF", "XGBoost"] if args.model == "Both" else [args.model]
    for model in models:
        filename = "FLU_T+1_to_T+4_" + ("RandomForest" if model == "RF" else "XGBoost") + ".ipynb"
        notebook = json.loads((root / filename).read_text(encoding="utf-8"))
        namespace = {"__name__": "__main__"}
        print(f"Preparing {model} ({args.mode})...", flush=True)
        # Cells after the main export are exploratory and not part of submission.
        for idx, cell in enumerate(notebook["cells"][:34]):
            if cell["cell_type"] != "code":
                continue
            if args.check_only and idx > 6:
                break
            source = "".join(cell["source"])
            exec(compile(source, f"{filename}:cell{idx}", "exec"), namespace)
        if args.check_only:
            print(f"{model}: input and date checks passed.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Preparation stopped: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
