Project Instructions
==============================

This repo contains the instructions for a machine learning project.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for describing highlights for using this ML project.
    ├── data
    │   └── raw            <- The original, immutable data dump.
    │       └── Phishing_Legitimate_full.csv    <- CSV file.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            
    │   └── README.md      <- Youtube Video Link.
    │   └── final_project_report <- final report .pdf format and supporting files.
    │   └── presentation   <-  final power point presentation.
    |
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`.
    │
    ├── src                <- Source code for use in this project.
       ├── __init__.py    <- Makes src a Python module.
       ├── main.py        <- Execute project.
       │
       ├── data           <- Scripts to download or generate data and pre-process the data.
       │   └── pre_processing.py           <- Class for data loading and pre processessing.
       │
       ├── models         <- Scripts to train models and then use trained models to make
       │   │                 predictions.
       │   ├── log_model.py          <- Class for logistic regression.
       │   └── tree_model.py          <- Class for decision tree.
       │
       └── visualization  <- Scripts to create exploratory and results oriented visualizations.
           └── visualize.py          <- Class for the visualizations.