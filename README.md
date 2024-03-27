# ML4SpatialAnalysis: A FMachine Learning Framework to do Spatial Analysis of Single Cell TNBC Data

This repository implements machine learning (ML) models in the context of spatial single-cell data analysis. It streamlines the entire process, from data preparation and model training to evaluation and interpretability.


## Getting Started

1. **Create a Conda Environment:**

```bash
conda env create -f requirements.yaml
```

## Running the Code

To run the code with specified hyperparameters in a config file:

```bash
python main.py configs/config.yaml
```

## Attribution Analysis

Perform attribution analysis of a pretrained model:

```bash
python run_attribution.py configs/config.yaml
```

## Code Repository Structure

1. **data/**: Contains all datasets within a folder.
2. **datautils/**: Handles preprocessing of cell table and graph computations:
    - `dataset.py`: Implements the dataset class.
    - `utils.py`: Contains basic utilities required for dataset construction.

3. **logmodels/**: Stores pretrained models.

4. **models/**: Contains all ML models and their training:
    - `factory.py`: Implements ML models.
    - `trainer.py`: Defines a class for feature preparation, training, and evaluation.
    - `attribution.py`: Analyzes weights/features for interpretability.

5. **utils/**: Holds common functions used in different parts of the code:
    - `utils.py`

6. **main.py**: Used for dispatching experiments.

    ./response_prediction.sh config/config.yaml model.name=xgboost model.fnorm=log1p

7. **run_attribution.py**: Dispatches attribution methods.
```
