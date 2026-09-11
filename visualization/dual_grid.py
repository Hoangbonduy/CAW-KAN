"""Plot initialized and trained dual-grid wavelets.

The current AdaptiveWaveletKAN implementation keeps ``a`` and ``b`` as
buffers. Training therefore changes the effective wavelet amplitude through
the CP factors A, B and D, but does not move or resize the grid itself.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import tempfile
import zipfile
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D


DATASETS = ("Etth1", "Etth2", "Ettm1", "Ettm2")
WAVELET_TYPES = ("mexican_hat", "morlet", "dog", "shannon")
WAVELET_COLORS = (
	"#0072B2",  # blue
	"#E69F00",  # orange
	"#009E73",  # green
	"#D55E00",  # vermilion
	"#CC79A7",  # pink
	"#56B4E9",  # sky blue
	"#F0E442",  # yellow
	"#000000",  # black
)


def find_bsdtar() -> str:
	candidates = [
		shutil.which("bsdtar"),
		"/home/hoang/miniconda3/bin/bsdtar",
		"/usr/bin/bsdtar",
		"/usr/local/bin/bsdtar",
	]
	for candidate in candidates:
		if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
			return candidate
	raise RuntimeError(
		"The attached checkpoint is an archive. Install bsdtar or add its directory to PATH."
	)


def compute_wavelet_response(z: torch.Tensor, wavelet_type: str) -> torch.Tensor:
	if wavelet_type == "mexican_hat":
		return (1.0 - z**2) * torch.exp(-0.5 * z**2)
	if wavelet_type == "morlet":
		return torch.cos(5.0 * z) * torch.exp(-0.5 * z**2)
	if wavelet_type == "dog":
		return z * torch.exp(-0.5 * z**2)
	if wavelet_type == "shannon":
		window = (z.abs() <= np.pi).to(z.dtype)
		return torch.sinc(z / np.pi) * window
	raise ValueError(f"Unsupported wavelet type: {wavelet_type}")


def make_initial_grid(num_wavelets: int, grid_size: float) -> tuple[torch.Tensor, torch.Tensor]:
	grid_min, grid_max = -grid_size, grid_size
	num_wide = (num_wavelets + 1) // 2
	num_narrow = num_wavelets - num_wide
	step = (grid_max - grid_min) / max(num_wide - 1, 1)

	wide_b = torch.linspace(grid_min, grid_max, num_wide)
	narrow_b = torch.linspace(grid_min + step / 2, grid_max - step / 2, num_narrow)
	wide_a = torch.ones(num_wide) * step * 2.42
	narrow_a = torch.ones(num_narrow) * step * 1.21
	return torch.cat((wide_b, narrow_b)), torch.cat((wide_a, narrow_a))


def _load_direct(path: str):
	try:
		return torch.load(path, map_location="cpu", weights_only=False)
	except Exception as direct_error:
		temp_dir = tempfile.mkdtemp(prefix="caw_kan_checkpoint_")
		try:
			bsdtar = find_bsdtar()
			subprocess.run(
				[bsdtar, "-xf", path, "-C", temp_dir],
				check=True,
				stdout=subprocess.DEVNULL,
				stderr=subprocess.PIPE,
				text=True,
			)
			members = subprocess.check_output([bsdtar, "-tf", path], text=True).splitlines()
			rebuilt = os.path.join(temp_dir, "checkpoint.zip")
			with zipfile.ZipFile(rebuilt, "w", zipfile.ZIP_STORED) as archive:
				for member in members:
					source = os.path.join(temp_dir, member)
					if os.path.isfile(source):
						archive.write(source, f"archive/{member}")
			return torch.load(rebuilt, map_location="cpu", weights_only=False)
		except Exception as archive_error:
			raise RuntimeError(f"Cannot read checkpoint {path}: {direct_error}; {archive_error}") from archive_error
		finally:
			shutil.rmtree(temp_dir, ignore_errors=True)


def load_state_dict(path: str) -> OrderedDict:
	state = _load_direct(path)
	if isinstance(state, dict) and "state_dict" in state:
		state = state["state_dict"]
	if not isinstance(state, dict):
		raise TypeError(f"Checkpoint {path} does not contain a state dict")
	return state


def collect_wavelet_tensors(state: dict, block_index: int = 0) -> dict[str, torch.Tensor]:
	prefix = f"blocks.{block_index}.adaptive_kan."
	tensors = {name: state[prefix + name].float() for name in ("A", "B", "D", "a", "b")}
	return tensors


def basis_curves(centers: torch.Tensor, widths: torch.Tensor, wavelet_type: str, x: torch.Tensor) -> torch.Tensor:
	z = (x[:, None] - centers[None, :]) / (widths.abs()[None, :] + 1e-6)
	return compute_wavelet_response(z, wavelet_type)


def plot_initialization(output_path: str, wavelet_type: str, grid_size: float, num_wavelets: int) -> None:
	centers, widths = make_initial_grid(num_wavelets, grid_size)
	num_wide = (num_wavelets + 1) // 2
	x = torch.linspace(float(centers.min() - widths.max()), float(centers.max() + widths.max()), 1200)
	curves = basis_curves(centers, widths, wavelet_type, x).numpy()

	fig, ax = plt.subplots(figsize=(12, 6.5))
	colors = WAVELET_COLORS[:num_wavelets]
	for index in range(num_wavelets):
		line_style = "-" if index < num_wide else "--"
		ax.plot(x.numpy(), curves[:, index], color=colors[index], lw=2, linestyle=line_style)
		ax.axvline(float(centers[index]), color=colors[index], alpha=0.18, lw=1)
	ax.set_title(f"Initialization: {wavelet_type} dual-grid wavelets")
	ax.set_xlabel("Input coordinate")
	ax.set_ylabel("Wavelet response")
	ax.grid(alpha=0.2)
	group_handles = [
		Line2D([0], [0], color="black", lw=2, linestyle="-", label="wide group"),
		Line2D([0], [0], color="black", lw=2, linestyle="--", label="narrow group"),
	]
	ax.legend(handles=group_handles, loc="upper right", frameon=True, framealpha=0.85)
	fig.tight_layout()
	fig.savefig(output_path, dpi=180)
	plt.close(fig)


def plot_trained(output_path: str, dataset: str, tensors: dict[str, torch.Tensor], wavelet_type: str) -> None:
	centers = tensors["b"][0, 0, 0]
	widths = tensors["a"][0, 0, 0]
	num_wavelets = centers.numel()
	num_wide = (num_wavelets + 1) // 2
	x = torch.linspace(float(centers.min() - widths.max()), float(centers.max() + widths.max()), 1200)
	curves = basis_curves(centers, widths, wavelet_type, x)

	# Collapse the learned CP tensor to one positive amplitude per wavelet.
	coefficients = torch.einsum("or,ir,wr->oiw", tensors["A"], tensors["B"], tensors["D"])
	amplitudes = coefficients.abs().mean(dim=(0, 1))
	relative = amplitudes / amplitudes.max().clamp_min(1e-12)

	fig, (ax_curve, ax_bar) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": (3, 1)})
	colors = WAVELET_COLORS[:num_wavelets]
	for index in range(num_wavelets):
		line_style = "-" if index < num_wide else "--"
		ax_curve.plot(
			x.numpy(),
			(curves[:, index] * relative[index]).numpy(),
			color=colors[index],
			lw=2,
			linestyle=line_style,
		)
	ax_curve.set_title(f"{dataset}: trained effective wavelets ({wavelet_type})")
	ax_curve.set_ylabel("Response x learned amplitude")
	ax_curve.grid(alpha=0.2)

	ax_bar.bar(np.arange(num_wavelets) + 1, relative.numpy(), color=colors)
	ax_bar.set_xlabel("Wavelet index")
	ax_bar.set_ylabel("Relative amplitude")
	ax_bar.set_xticks(np.arange(num_wavelets) + 1)
	ax_bar.set_ylim(0, 1.08)
	ax_bar.grid(axis="y", alpha=0.2)
	group_handles = [
		Line2D([0], [0], color="black", lw=2, linestyle="-", label="wide group"),
		Line2D([0], [0], color="black", lw=2, linestyle="--", label="narrow group"),
	]
	ax_curve.legend(handles=group_handles, loc="upper right", frameon=True, framealpha=0.85)
	fig.tight_layout()
	fig.savefig(output_path, dpi=180)
	plt.close(fig)


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--project-root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	parser.add_argument("--output-dir", default=None)
	parser.add_argument("--wavelet-type", choices=WAVELET_TYPES, default="mexican_hat")
	parser.add_argument("--grid-size", type=float, default=3.0)
	parser.add_argument("--num-wavelets", type=int, default=8)
	args = parser.parse_args()

	output_dir = args.output_dir or os.path.join(args.project_root, "visualization_results")
	os.makedirs(output_dir, exist_ok=True)
	plot_initialization(os.path.join(output_dir, "01_initialization.png"), args.wavelet_type, args.grid_size, args.num_wavelets)

	for index, dataset in enumerate(DATASETS, start=2):
		checkpoint = os.path.join(args.project_root, "checkpoints", f"{dataset}_512_96", "checkpoint.pth")
		output = os.path.join(output_dir, f"{index:02d}_{dataset}_trained.png")
		try:
			tensors = collect_wavelet_tensors(load_state_dict(checkpoint))
			plot_trained(output, dataset, tensors, args.wavelet_type)
			print(f"saved {output}")
		except Exception as error:
			print(f"ERROR {dataset}: {error}")


if __name__ == "__main__":
	main()
