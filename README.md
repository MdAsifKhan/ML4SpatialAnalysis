# ML4SpatialAnalysis: A Machine Learning Framework to do Spatial Analysis of Single Cell TNBC Data

This repository implements machine learning (ML) models in the context of spatial single-cell data analysis. It streamlines the entire process, from data preparation and model training to evaluation and interpretability.


## Getting Started

1. **Create a Conda Environment:**

```bash
conda env create -f requirements.yaml
```

## Running the Code

Use the conda environment from `requirements.yaml` (python 3.8, torch 1.12, torch_geometric 2.6). ROI-level GNN or classical model, one run per config:

```bash
python main.py --config configs/config.yaml
```

`main.py` takes only `--config`; to sweep settings write one yaml per cell with `scripts/make_variant_configs.py` (gconv x gmethod x k x node_f x seed, plus a manifest for `scripts/run_manifest_cpu.sbatch`) and summarise the metric sinks with `scripts/collect_variant_results.py`.

Patient-level benchmark (package `patientgnn/`, config `configs/patient_benchmark.yaml`):

```bash
# Build the versioned QC cache and run all patient-level comparisons.
python -m patientgnn.run_patient_benchmark \
  --config configs/patient_benchmark.yaml \
  --build-cache

bash scripts/submit_regression.sh   # slurm: cache -> classical baselines -> DeepSets/GNN
```

Compare `expressions`, `celltypes`, and `combined` inputs for:

- Regularized logistic regression on patient-level ROI summaries.
- Random forest and XGBoost on exactly the same patient-level summaries and
  outer folds, matching the classical models used in the workshop analysis.
- A spatial logistic baseline with abundance-normalized cell-type contacts.
- Hierarchical DeepSets (same encoders and pooling as the GNN, no edges).
- A two-layer residual GINE using physical contact distances.
- An edge-shuffled negative control for the combined GNN.

## Data

`dataset.DATA_PATH` holds `<cell_filename>.csv` (one row per cell: `fov`, `cell_meta_cluster`, `centroid-0`, `centroid-1`, the marker columns, optionally `qc_pass`) and `<response_filename>.csv`. The loader expects the `CleanCohort_Metadata` schema: `LEAP_ID`, `Sample_Type_(pre/post treatment)`, `NACT_Treatment _Group`, `Response`, `Patient_ID`, `Stain_Batch`. The older `processed_response_RCB` file is not supported. `configs/CleanCohort_Metadata.csv` is the manuscript cohort (63 patients, 37 responders / 26 non-responders); the copy in the shared data directory is a later revision with different labels, so pin the one you mean in `response_filename`.

## Attribution Analysis

Evaluate and attribute a saved run. `main.py` stamps run names with `time.time()`; the newest matching model is used unless `--run <timestamp>` is given:

```bash
python run_evaluation.py --config configs/config.yaml [--run 1753249753.4735453]
```

## Code Repository Structure

1. **data/**: Contains all datasets within a folder.
2. **datautils/**: Handles preprocessing of cell table and graph computations:
    - `dataset.py`: Implements the dataset class.
    - `utils.py`: Contains basic utilities required for dataset construction.

3. **logmodels/**: Stores pretrained models.

4. **models/**: Contains all ML models and their training:
    - `graph_networks.py`, `gcn.py`: GNN architectures and their training wrapper.
    - `trainer.py`: Defines a class for feature preparation, training, and evaluation.
    - `evaluation.py`: Analyzes weights/features for interpretability.

5. **mainutils/**: Graph construction, splits, metrics (`utils.py`) and train-fit marker normalisation (`normalization.py`).

6. **patientgnn/**: Patient-level benchmark (cohort cache, classical baselines, DeepSets/GINE).

7. **main.py**: Used for dispatching experiments.

8. **run_evaluation.py**: Dispatches attribution methods on a saved run.
```
