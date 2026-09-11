"""Calculate mean +/- standard deviation across seeds for forecasting results."""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path
import statistics


RESULT_PATTERN = re.compile(
	r"^(?P<pred_len>\d+):\s*mse=(?P<mse>[-+0-9.eE]+),\s*mae=(?P<mae>[-+0-9.eE]+)$"
)


def parse_result_file(path: Path) -> dict[tuple[str, str, int], dict[str, float]]:
	"""Parse one seed result file into model, dataset, and prediction-length keys."""
	results: dict[tuple[str, str, int], dict[str, float]] = {}
	model = None
	dataset = None

	for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
		line = raw_line.strip()
		if not line or line.startswith("seed="):
			continue

		match = RESULT_PATTERN.match(line)
		if match:
			if model is None or dataset is None:
				raise ValueError(f"Missing model/dataset before {path}:{line_number}")
			key = (model, dataset, int(match.group("pred_len")))
			results[key] = {
				"mse": float(match.group("mse")),
				"mae": float(match.group("mae")),
			}
			continue

		if line in {"TimeKAN", "CAW_KAN", "DLinear"}:
			model = line
			dataset = None
		else:
			dataset = line

	return results


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("Cannot calculate statistics for an empty list")
    if len(values) == 1:
        return values[0], 0.0
        
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    return mean, std


def format_stat(values: list[float]) -> str:
	mean, std = mean_std(values)
	return f"{mean:.4f} +- {std:.4f}"


def calculate_statistics(paths: list[Path]) -> dict[tuple[str, str, int], dict[str, list[float]]]:
	values: dict[tuple[str, str, int], dict[str, list[float]]] = defaultdict(
		lambda: {"mse": [], "mae": []}
	)

	for path in paths:
		for key, metrics in parse_result_file(path).items():
			values[key]["mse"].append(metrics["mse"])
			values[key]["mae"].append(metrics["mae"])

	expected_seeds = len(paths)
	incomplete = {
		key: metrics
		for key, metrics in values.items()
		if len(metrics["mse"]) != expected_seeds
		or len(metrics["mae"]) != expected_seeds
	}
	if incomplete:
		missing = ", ".join(
			f"{model}/{dataset}/{pred_len} ({len(metrics['mse'])}/{expected_seeds})"
			for (model, dataset, pred_len), metrics in incomplete.items()
		)
		raise ValueError(f"Some result groups do not contain every seed: {missing}")

	return values


def format_statistics(values: dict[tuple[str, str, int], dict[str, list[float]]]) -> str:
	current_model = None
	current_dataset = None
	lines = []

	for model, dataset, pred_len in sorted(values):
		if model != current_model:
			lines.append(f"\n{model}")
			current_model = model
			current_dataset = None
		if dataset != current_dataset:
			lines.append(dataset)
			current_dataset = dataset

		metrics = values[(model, dataset, pred_len)]
		lines.append(
			f"{pred_len}: mse={format_stat(metrics['mse'])}, "
			f"mae={format_stat(metrics['mae'])}"
		)

	return "\n".join(lines).lstrip()


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Calculate mean +- population std across result files."
	)
	parser.add_argument(
		"files",
		nargs="*",
		type=Path,
		help="Result files, one file per seed (default: TimeKAN2021/2022/2023.txt).",
	)
	args = parser.parse_args()

	base_dir = Path(__file__).resolve().parent
	paths = args.files or [base_dir / f"TimeKAN{seed}.txt" for seed in (2021, 2022, 2023)]
	values = calculate_statistics(paths)
	output = format_statistics(values)
	output_path = base_dir / "full3seed.txt"
	output_path.write_text(output + "\n")
	print(output)


if __name__ == "__main__":
	main()
