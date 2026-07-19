# mlops_eurosat

MLOps pipeline for land use classification on the [EuroSAT dataset](https://github.com/phelber/EuroSAT).

The project trains a convolutional neural network to classify Sentinel-2 satellite images
into ten land use / land cover classes, and wraps the model in a full MLOps stack.

## Components

- **Data versioning** with DVC
- **Training** with PyTorch Lightning, configured via Hydra ([configs/](https://github.com/mlops-eurosat/MLOps-EuroSAT/tree/main/configs))
- **Experiment tracking & sweeps** with Weights & Biases
- **Model registry** with automatic deployment triggers
- **Serving** via a FastAPI inference API and a Streamlit frontend
- **Cloud** training and pipelines on Google Cloud Vertex AI
- **CI/CD** with GitHub Actions, CML data reports, and pre-commit hooks
- **Monitoring** including data drift detection with Evidently

## Getting started

```bash
git clone https://github.com/mlops-eurosat/MLOps-EuroSAT.git
cd MLOps-EuroSAT
pip install -r requirements.txt -r requirements_dev.txt -e .
```

Common tasks are available through [Invoke](https://www.pyinvoke.org/):

```bash
invoke --list        # show all tasks
invoke train         # train the model
invoke test          # run the test suite
invoke serve-docs    # serve this documentation locally
```

See the [Python Module Reference](api.md) for documentation of the package's classes and functions.
