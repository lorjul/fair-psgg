# A Fair Ranking and New Model for Panoptic Scene Graph Generation

**We are currently preparing this repo for the camera ready version. The current code may still contain some rough edges.**

This is the official implementation of our paper "A Fair Ranking and New Model for Panoptic Scene Graph Generation", accepted at ECCV 2024. For more information, please visit the [project page](https://lorjul.github.io/fair-psgg/).

## Setup

### Environment

Install the [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer) package manager. In the folder that contains this README, run `poetry install`.

Alternatively, you can use pip to install the different packages manually. However, using Poetry is recommended.

### Data

<!-- direct link: https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/EgQzvsYo3t9BpxgMZ6VHaEMBDAb7v0UgI8iIAExQUJq62Q?e=fIY3zh -->

Download the PSG datasets from [here](https://github.com/Jingkang50/OpenPSG?tab=readme-ov-file#updates) and extract them.

## Usage

### Training

To train a model, use the following code:

``` sh
# first, activate the environment (if not done)
poetry shell
# then, run the training command
fair-psgg \
    --anno /path/to/psg/psg/psg.json \
    --img /path/to/psg/coco \
    --seg /path/to/psg/coco \
    --epochs 40 \
    ./configs/table2/masks-loc-sem.json \
    /output/path/masks-loc-sem
```

For more information, run `fair-psgg --help`.

The provided config files are grouped by the supplementary tables.

### Inference

``` sh
poetry shell
python -m fair_psgg.tasks.inference \
    /path/to/psg/psg/psg.json \
    /path/to/psg/coco \
    /path/to/psg/coco \
    /path/to/model/output/folder \
    --split test
```

## Citation

If you found our paper or code helpful, please consider citing it:

``` bibtex
@misc{lorenz2024fairpsgg,
    title={A Fair Ranking and New Model for Panoptic Scene Graph Generation}, 
    author={Julian Lorenz and Alexander Pest and Daniel Kienzle and Katja Ludwig and Rainer Lienhart},
    year={2024},
    eprint={2407.09216},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2407.09216}, 
}
```
