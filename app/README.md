# Important

The files in this directory were created to work with the uv python project manager. It will also work with normal
python and conda environments, but it is up to you to ensure all dependencies are installed and also the correct python
version is used. The python version as well as the dependencies can be found in the ```pyproject.toml``` file.

# Testing yourself on the dataset

We have also created an app where you can label parts of the dataset yourself and see how aligned you are with the
researchers. You can run the app by executing the following command:
```
uv run dataset_labeler.py --dataset-path=<PATH_TO_MMHS150K>/MMHS150K_GT.json --img-path=<PATH_TO_MMHS150K>/img_resized --img-text-path=<PATH_TO_MMHS150K>/img_txt --output-path=./annotations.jsonl --limit=300
```

This can then be run in conjunction with ```check_annotations.py``` to compute some metrics based on your annotations.
This file can also be used with any of the other models in this repository that output a single hatescore between 0-4.
Run this file by executing the following command:
```
uv run check_annotations.py --dataset-path=<PATH_TO_MMHS150K>/MMHS150K_GT.json --annotation-path=<PATH_TO_ANNOTATIONS> --graph-name=<NAME_OF_GRAPH> --split=<PATH_TO_MMHS150K>/splits
```