# LE-EpiGNODE

## Directory structure

```text
LE-EpiGNODE/
  train.py                            # Train a selected experiment
  predict.py                          # Checkpoint prediction
  configs/training/<synthetic|real>/   # Scenario training settings
  code/models/                        # Three synthetic and three real models
  code/experiments/<synthetic|real>/   # Scenario training entry points
  code/inference/                     # Data loading, prediction and plotting
  code/training/                      # Training outputs and checkpoints
  data/<synthetic|real>/<ID>/          # Input datasets; synthetic generators
  results/<synthetic|real>/<ID>/       # Model weights and reference results
  docs/                               # Data fields and scenario index
  environment.yml                     # Conda environment
  requirements.txt                    # Python dependencies
  CITATION.cff                        # Citation metadata
```

There are 14 synthetic scenarios and three real scenarios: `ontario`, `hubei`, and `spain`.

## Experiment environment

| Component | Setting |
|---|---|
| Operating system | Windows |
| Conda environment | DL |
| Python | 3.8.10 |
| PyTorch | 2.4.1+cu124 |
| GPU | Two NVIDIA GeForce RTX 3090 GPUs, 24 GiB each |

On Windows, run the installation and activation commands in Anaconda Prompt.

```bash
conda env create -f environment.yml
conda activate DL
python -m pip install -r requirements.txt
```

If the `DL` environment already exists, activate it and run the dependency installation command.
Use `cuda:0`, `cuda:1`, or `cpu` as the device.

## Usage

Run commands from the package directory.

### List scenarios

```bash
python predict.py --list
python train.py --list
```

### Predict with the provided checkpoints

```bash
python predict.py --scenario spain --device cuda:0 --output ../spain_predictions
python predict.py --scenario all --device cuda:0 --output ../predictions
```

Outputs are saved in `<output>/<synthetic|real>/<ID>/`. The default root is `../predictions/`.
Existing outputs are refused unless `--overwrite` is specified. This option replaces only the selected scenario directories.

Each output includes `pred_u.npy` (time x node x 1), `pred_u_city.npy` (time x city x 1), `GE_city.npy`, `GL_age.npy`, `case_forecast.png`, and `city_forecast.png`.
The two case arrays contain gammaI. Multicity scenarios also output `pred_mobility.npy` and `mobility_forecast.png`.

### Train a model

```bash
python train.py --run spain --device cuda:0
python train.py --run lagrangian_BMCM --device cuda:1 --output ../training_runs
```

The default root is `../training_runs/`. Each scenario has its own working directory.
Experiment outputs are in `<output>/<ID>/experiments/<dataset>/exp_<time>_<identifier>/`:

```text
config.yaml
training.log
loss_history.csv
checkpoints/best_model.pth
predictions/                     # The same arrays and figures as checkpoint prediction
```

The best checkpoint is selected using the minimum monitoring loss before the parameter update.
Checkpoint `epoch` counts completed updates; the first candidate is epoch 0.
Use `--epochs <count>` to specify a different training length.

To prepare a working directory for manual execution:

```bash
python train.py --prepare spain --output ../manual_runs
cd ../manual_runs/spain
python main.py --help
python main.py --device cuda:0
```

Return to the package directory before running the commands below.

### Generate synthetic data

```bash
python data/synthetic/lagrangian_BMCM/generate_SIR_data.py --output ../generated_data/lagrangian_BMCM
```
