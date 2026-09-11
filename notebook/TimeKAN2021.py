import json
import re
from pathlib import Path


OUTPUT_NAME = "TimeKAN2021.txt"
DATASET_ORDER = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Exchange"]
HORIZON_ORDER = ["96", "192", "336", "720"]

MODEL_CONFIGS = [
	{
		"name": "TimeKAN",
		"notebook": "",
		"mode": "manual",
	},
	{
		"name": "CAW_KAN",
		"notebook": "CAW_KAN_original2021seed.ipynb",
		"mode": "single_horizon_per_cell",
	},
	{
		"name": "DLinear",
		"notebook": "ICLR_DLinear_seed2021original.ipynb",
		"mode": "multi_horizon_per_cell",
	},
]

TIMEKAN_MANUAL_MSE = {
	"ETTh1": {"96": "0.397", "192": "0.415", "336": "0.434", "720": "0.457"},
	"ETTh2": {"96": "0.343", "192": "0.382", "336": "0.432", "720": "0.455"},
	"ETTm1": {"96": "0.351", "192": "0.372", "336": "0.397", "720": "0.427"},
	"ETTm2": {"96": "0.257", "192": "0.298", "336": "0.329", "720": "0.395"},
	"Exchange": {"96": "0.215", "192": "0.310", "336": "0.441", "720": "0.734"},
}

TIMEKAN_MANUAL_MAE = {
	"ETTh1": {"96": "0.397", "192": "0.415", "336": "0.434", "720": "0.457"},
	"ETTh2": {"96": "0.343", "192": "0.382", "336": "0.432", "720": "0.455"},
	"ETTm1": {"96": "0.351", "192": "0.372", "336": "0.397", "720": "0.427"},
	"ETTm2": {"96": "0.257", "192": "0.298", "336": "0.329", "720": "0.395"},
	"Exchange": {"96": "0.215", "192": "0.310", "336": "0.441", "720": "0.734"},
}

SCRIPT_PATTERN = re.compile(
	r"scripts/(?P<dataset>ETTh1|ETTh2|ETTm1|ETTm2|Exchange)/"
	r"(?P<script>[A-Za-z0-9_]+)_(?P<horizon>96|192|336|720)\.sh"
)
SCRIPT_PATTERN_LOWER = re.compile(
	r"scripts/(?P<dataset>Etth1|Etth2|Ettm1|Ettm2|Exchange)/"
	r"(?P<script>[A-Za-z0-9_]+)_(?P<horizon>96|192|336|720)\.sh"
)
SCRIPT_DLINEAR_DATASET_PATTERN = re.compile(
	r"scripts/EXP-LongForecasting/DLinear/(?P<script>etth1|etth2|ettm1|ettm2|exchange_rate)\.sh"
)
METRIC_PATTERN = re.compile(
	r"mse:\s*(?P<mse>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s*"
	r"mae:\s*(?P<mae>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)

DATASET_NAME_MAP = {
	"ETTh1": "ETTh1",
	"ETTh2": "ETTh2",
	"ETTm1": "ETTm1",
	"ETTm2": "ETTm2",
	"Exchange": "Exchange",
	"Etth1": "ETTh1",
	"Etth2": "ETTh2",
	"Ettm1": "ETTm1",
	"Ettm2": "ETTm2",
	"exchange_rate": "Exchange",
	"etth1": "ETTh1",
	"etth2": "ETTh2",
	"ettm1": "ETTm1",
	"ettm2": "ETTm2",
}


def _to_text(value):
	if isinstance(value, list):
		return "".join(value)
	return str(value)


def _collect_output_text(cell):
	outputs = cell.get("outputs", [])
	output_text_chunks = []
	for out in outputs:
		for key in ("text", "traceback"):
			if key in out:
				output_text_chunks.append(_to_text(out[key]))
		data = out.get("data", {}) if isinstance(out, dict) else {}
		for key in ("text/plain", "text"):
			if key in data:
				output_text_chunks.append(_to_text(data[key]))
	return "\n".join(output_text_chunks)


def extract_single_horizon_metrics(notebook_path: Path):
	with notebook_path.open("r", encoding="utf-8") as f:
		notebook = json.load(f)

	results = {dataset: {} for dataset in DATASET_ORDER}

	for cell in notebook.get("cells", []):
		if cell.get("cell_type") != "code":
			continue

		source_text = _to_text(cell.get("source", ""))
		script_match = SCRIPT_PATTERN.search(source_text)
		if not script_match:
			script_match = SCRIPT_PATTERN_LOWER.search(source_text)
		if not script_match:
			continue

		dataset = DATASET_NAME_MAP[script_match.group("dataset")]
		horizon = script_match.group("horizon")

		output_text = _collect_output_text(cell)
		metric_match = METRIC_PATTERN.search(output_text)
		if metric_match:
			results[dataset][horizon] = {
				"mse": metric_match.group("mse"),
				"mae": metric_match.group("mae"),
			}

	return results


def extract_multi_horizon_metrics(notebook_path: Path):
	with notebook_path.open("r", encoding="utf-8") as f:
		notebook = json.load(f)

	results = {dataset: {} for dataset in DATASET_ORDER}

	for cell in notebook.get("cells", []):
		if cell.get("cell_type") != "code":
			continue

		source_text = _to_text(cell.get("source", ""))
		script_match = SCRIPT_DLINEAR_DATASET_PATTERN.search(source_text)
		if not script_match:
			continue

		dataset = DATASET_NAME_MAP[script_match.group("script")]
		output_text = _collect_output_text(cell)
		metric_matches = list(METRIC_PATTERN.finditer(output_text))

		for horizon, metric_match in zip(HORIZON_ORDER, metric_matches):
			results[dataset][horizon] = {
				"mse": metric_match.group("mse"),
				"mae": metric_match.group("mae"),
			}

	return results


def extract_metrics_by_model(notebook_dir: Path):
	all_results = {}
	for config in MODEL_CONFIGS:
		if config["mode"] == "manual" and config["name"] == "TimeKAN":
			manual_results = {dataset: {} for dataset in DATASET_ORDER}
			for dataset in DATASET_ORDER:
				for horizon in HORIZON_ORDER:
					mse_value = TIMEKAN_MANUAL_MSE.get(dataset, {}).get(horizon)
					mae_value = TIMEKAN_MANUAL_MAE.get(dataset, {}).get(horizon)
					if mse_value is not None or mae_value is not None:
						manual_results[dataset][horizon] = {
							"mse": mse_value if mse_value is not None else "N/A",
							"mae": mae_value if mae_value is not None else "N/A",
						}
			all_results[config["name"]] = manual_results
			continue

		notebook_path = notebook_dir / config["notebook"]
		if not notebook_path.exists():
			all_results[config["name"]] = {dataset: {} for dataset in DATASET_ORDER}
			continue

		if config["mode"] == "single_horizon_per_cell":
			model_results = extract_single_horizon_metrics(notebook_path)
		else:
			model_results = extract_multi_horizon_metrics(notebook_path)

		all_results[config["name"]] = model_results

	return all_results


def format_results(all_results):
	lines = []
	for model_name in [cfg["name"] for cfg in MODEL_CONFIGS]:
		lines.append(model_name)
		model_results = all_results.get(model_name, {})
		for dataset in DATASET_ORDER:
			lines.append(dataset)
			for horizon in HORIZON_ORDER:
				item = model_results.get(dataset, {}).get(horizon)
				if item:
					lines.append(f"{horizon}: mse={item['mse']}, mae={item['mae']}")
				else:
					lines.append(f"{horizon}: N/A")
			lines.append("")
		lines.append("")
	return "\n".join(lines).rstrip() + "\n"


def main():
	current_dir = Path(__file__).resolve().parent
	output_path = current_dir / OUTPUT_NAME

	all_results = extract_metrics_by_model(current_dir)
	content = format_results(all_results)
	output_path.write_text(content, encoding="utf-8")

	print(content)
	print(f"Saved to: {output_path}")


if __name__ == "__main__":
	main()
