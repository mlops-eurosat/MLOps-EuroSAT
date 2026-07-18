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

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).

## Data attribution

This project uses the [EuroSAT dataset](https://github.com/phelber/EuroSAT)
(Helber et al.), released under the MIT License. EuroSAT contains modified
Copernicus Sentinel data (2017), used in accordance with the
[Copernicus Sentinel Data Terms and Conditions](https://sentinel.esa.int/documents/247904/690755/Sentinel_Data_Legal_Notice).

If you use this work, please cite:

> Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). EuroSAT: A novel
> dataset and deep learning benchmark for land use and land cover
> classification. *IEEE Journal of Selected Topics in Applied Earth
> Observations and Remote Sensing*, 12(7), 2217–2226.


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

