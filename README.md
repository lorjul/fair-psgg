# Fair PSGG

## Setup

### Environment

Install the [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer) package manager. In the folder that contains this README, run `poetry install`.

Alternatively, you can use pip to install the different packages manually. However, using Poetry is recommended.

### Data

<!--https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/EgQzvsYo3t9BpxgMZ6VHaEMBDAb7v0UgI8iIAExQUJq62Q?e=fIY3zh-->

Download the PSG datasets from [here](https://github.com/Jingkang50/OpenPSG?tab=readme-ov-file#updates) and extract them.

## Usage

### Training

To train a model, use the following code:

``` sh
# first, activate the environment (if not done)
poetry shell
# run the training command
python -m fair_psgg --help
```
