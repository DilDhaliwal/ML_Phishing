[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-8d59dc4de5201274e310e4c54b9627a8934c3b88527886e3b421487c677d23eb.svg)](https://classroom.github.com/a/YCTbQ0qx)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10581728&assignment_repo_type=AssignmentRepo)
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
           └── visualize.py          <- Class for visualizations.