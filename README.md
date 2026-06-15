# mlops_eurosat

MLOps pipeline for land use classification on the EuroSAT dataset.

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       ├── cml_data.yml
│       ├── linting.yaml
│       └── tests.yaml
├── configs/                  # Configuration files
│   ├── model/
│   ├── training/
│   ├── wandb/
│   ├── config.yaml
│   └── sweep.yaml
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.dockerfile
│   ├── train.dockerfile
│   └── trigger.dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yaml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── mlops_eurosat/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── dataset_statistics.py
│   │   ├── evaluate.py
│   │   ├── frontend.py
│   │   ├── model.py
│   │   ├── pipeline.py
│   │   ├── registry_trigger.py
│   │   ├── train.py
│   │   ├── vertex_registry.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   ├── test_evaluate.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
