"""Uruchom: python -m experiments.bike_sharing.run --download"""

import argparse
import json
from pathlib import Path
import sys

from experiments.bike_sharing.artifacts import export_study
from src.analysis_validation import AnalysisError
from src.bike_sharing_data import DATA_DIR, download_bike_sharing, load_bike_sharing
from src.bike_sharing_study import CONFIG_PATH, StudyConfig, run_bike_study


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Powtarzalne badanie oficjalnego hour.csv z UCI 275.")
    parser.add_argument("--download", action="store_true", help="Pobierz oficjalny zbiór, jeśli jeszcze go nie ma.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent, help="Katalog REPORT.md i results/.")
    args = parser.parse_args(argv)
    try:
        config = StudyConfig.load(args.config)
        if args.download:
            download_bike_sharing()
        data = load_bike_sharing()
        provenance = json.loads((DATA_DIR / "manifest.json").read_text())
        result = run_bike_study(data, config)
        export_study(result, provenance, args.output)
    except (AnalysisError, OSError) as exc:
        print(f"Badanie nie zostało ukończone: {exc}", file=sys.stderr)
        return 1
    print(result.tables["regression_metrics"].to_string())
    print("Wybrane k:", result.metadata["selected_k"])
    print("Wariancja zachowana przez PC1/PC2:", result.metadata["pca_retained_variance"])
    print("Raport:", args.output / "REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
