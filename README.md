## Requirements

```bash
cd CAW-KAN
conda create -n cawkan python=3.12
conda activate cawkan
pip install -r requirements.txt
```

## Data Preparation

We use the datasets provided by the [Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) repository for our experiments. 

Once you have downloaded the datasets, please create a `dataset` folder in the root directory of the project and put all the data files inside it.

## Repository Structure

- **`models/`**: Contains the main network architecture definition (`CAW_KAN.py`).
- **`layers/`**: Contains the component modules of the model such as `WaveletKAN.py`, `Embed.py`, and `StandardNorm.py`.
- **`Architectural components/`**: Contains model architecture variants used for ablation studies:
  - `CAW_KAN_MLP.py`: Replaces WaveletKAN with an MLP (uses `StandardMLPLayer.py` instead of `WaveletKAN.py`).
  - `CAW_KAN_without_Conv1d.py`: Removes the Conv1D layer.
  - `CAW_KAN_without_residual_connections.py`: Removes residual connections.
- **`Grid & Factorization/`**: Contains layer variants for ablation studies (all configurations default to the Mexican-hat wavelet):
  - `Diagonal_only.py`: No mixing between latent features.
  - `StandardMLPLayer.py`: Uses an MLP instead of WaveletKAN.
  - `WaveletKAN_Dual_grid.py`: Full tensor with a dual-grid, no CP factorization applied.
  - `WaveletKAN_Single_grid.py`: Full tensor with a single-grid, no CP factorization applied.
  - `WaveletKAN_Single_grid_CP.py`: Single-grid with CP factorization.
- **`exp/`**: Contains the classes responsible for handling the training and evaluation processes (e.g., `exp_long_term_forecasting.py`).
- **`scripts/`**: Contains execution scripts to run the model on various datasets for different prediction lengths (`pred_len` of 96, 192, 336, and 720).
- **`experiment_scripts/`**: Contains scripts for parameter sensitivity analysis and ablation studies. Specifically, it includes:
  - `grid_size`: Analyzes the `grid_size` parameter with values in [1.5, 3.0, 4.5, 6.0, 7.5].
  - `kernel_size`: Analyzes the `kernel_size` parameter in the Conv1D layer with values in [1, 3, 7, 15].
  - `num_blocks`: Analyzes the `num_blocks` (number of blocks) parameter with values in [1, 2, 3, 4].
  - `num_wavelets_test`: Analyzes the number of wavelets parameter with values in [4, 6, 8, 10, 12].
  - `wavelets_type`: Analyzes different wavelet types, including Mexican_hat, DoG, and Morlet.

## Reproducing Results

- **Main Results:** To run the model on the Etth1 dataset with a prediction length of 96, execute:
  ```bash
  bash scripts/Etth1/etth1_96.sh
  ```
- **Parameter Sensitivity Results:** To explore different parameters, such as analyzing the kernel size of the Conv1D layer for the Etth1 dataset with a prediction length of 96, run:
  ```bash
  bash experiment_scripts/kernel_size/Etth1/etth1_96.sh
  ```
- **Different Structural Results:** To test different architectures, replace the original files in the `models/` or `layers/` directories with the corresponding variants from `Architectural components/` or `Grid & Factorization/`, and then run the execution scripts as shown above.
- **Loss Function Variation:** You can change the loss function for appropriate scenarios by modifying the `_select_criterion` method in the experimental scripts.

## Acknowledgements

We appreciate the following github repositories provided valuable code bases and datasets for our work:
- [Time-Series-Library](https://github.com/thuml/Time-Series-Library)
- [Wav-KAN](https://github.com/zavareh1/Wav-KAN)