# mlops_eurosat

[![Unit Tests](https://github.com/mlops-eurosat/MLOps-EuroSAT/actions/workflows/tests.yaml/badge.svg?branch=main)](https://github.com/mlops-eurosat/MLOps-EuroSAT/actions/workflows/tests.yaml)
[![Code linting](https://github.com/mlops-eurosat/MLOps-EuroSAT/actions/workflows/linting.yaml/badge.svg?branch=main)](https://github.com/mlops-eurosat/MLOps-EuroSAT/actions/workflows/linting.yaml)
[![Deploy Docs](https://github.com/mlops-eurosat/MLOps-EuroSAT/actions/workflows/docs.yaml/badge.svg?branch=main)](https://github.com/mlops-eurosat/MLOps-EuroSAT/actions/workflows/docs.yaml)

MLOps pipeline for land use classification on the EuroSAT dataset.

Documentation: <https://mlops-eurosat.github.io/MLOps-EuroSAT/>

## Project structure

The directory structure of the project looks like this:
```txt
├── .devcontainer/            # VS Code devcontainer
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       ├── cml_data.yml
│       ├── docs.yaml
│       ├── linting.yaml
│       └── tests.yaml
├── configs/                  # Hydra configuration
│   ├── model/
│   ├── training/
│   ├── wandb/
│   ├── config.yaml
│   └── sweep.yaml
├── data/                     # DVC-tracked data
├── dockerfiles/              # One image per service
│   ├── api.dockerfile
│   ├── frontend.dockerfile
│   ├── monitoring.dockerfile
│   ├── train.dockerfile
│   └── trigger.dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yaml
│   └── source/
├── models/                   # Trained models
├── reports/                  # Course report and figures
├── src/                      # Source code
│   └── mlops_eurosat/
│       ├── api.py
│       ├── data.py
│       ├── data_drift.py
│       ├── dataset_statistics.py
│       ├── evaluate.py
│       ├── frontend_map.py
│       ├── model.py
│       ├── model_registry.py
│       ├── monitoring_api.py
│       ├── pipeline.py
│       ├── profiling.py
│       ├── quantize.py
│       ├── registry_trigger.py
│       ├── train.py
│       └── visualize.py
├── tests/                    # Unit tests
│   └── performancetests/     # Locust load test
├── .pre-commit-config.yaml
├── cloudbuild.yaml
├── dvc.yaml
├── LICENSE
├── pyproject.toml
├── README.md
├── requirements.txt          # Base requirements
├── requirements_*.txt        # Dev and per-service requirements
├── tasks.py                  # Invoke tasks
└── vertex_config_*.yaml      # Vertex AI job configs
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).

## Attributions

### Data

This project uses the [EuroSAT dataset](https://github.com/phelber/EuroSAT)
(Helber et al.), released under the MIT License. EuroSAT contains modified
Copernicus Sentinel data (2017).

If you use this work, please cite:

> Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). EuroSAT: A novel
> dataset and deep learning benchmark for land use and land cover
> classification. *IEEE Journal of Selected Topics in Applied Earth
> Observations and Remote Sensing*, 12(7), 2217–2226.

### Template

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps), developed for the
[DTU MLOps course](https://github.com/SkafteNicki/dtu_mlops):

```bibtex
@misc{skafte_mlops,
    author       = {Nicki Skafte Detlefsen},
    title        = {Machine Learning Operations},
    howpublished = {\url{https://github.com/SkafteNicki/dtu_mlops}},
    year         = {2026}
}
```
