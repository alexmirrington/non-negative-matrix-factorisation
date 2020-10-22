# Non-negative Matrix Factorisation

<p align="center">
  <a href="https://github.com/alexmirrington/non-negative-matrix-factorisation/actions?query=workflow%3Astyle">
    <img
      src="https://github.com/alexmirrington/non-negative-matrix-factorisation/workflows/style/badge.svg"
      alt="Code style check"
    />
  </a>
  <a href="https://github.com/psf/black">
    <img
      src="https://img.shields.io/badge/code%20style-black-000000.svg"
      alt="Code style: black"
    />
  </a>
</p>

----------------------

## Prerequisites

```Text
python >= 3.8
```

We used the following pip package versions, though any up-to-date versions should be fine:

```Text
numpy==1.19.2
Pillow==8.0.0
sklearn==0.23.2
pre-commit==2.7.1
wandb==0.10.5
termcolor==1.1.0
```

## Getting Started

Make sure to install all python requirements like so:

`pip install -r requirements.txt`

Download the ORL and YaleB datasets from Canvas, and place then in the `code/data` folder. After unzipping the datasets, the `code/data` should contain a `README` file, along with a folder called `CroppedYaleB` containing the YaleB dataset and a folder called `ORL` containing the ORL dataset.

By default, `code/algorithm/main.py` will run a standard NMF model on a 90% subset of the ORL dataset, with no additional noise applied to the training data. Results will be logged to the terminal as well as to a `.jsonl` file in `code/algorithm/results`.

### Changing the noise

To change the type of noise used, you can specify the `--noise` parameter along with any additional noise parameters required for the chosen noise type:

**Uniform noise:**  `python code/algorithm/main.py --noise uniform --noise_mean 0 --noise_std 0.1`

**Gaussian noise:** `python code/algorithm/main.py --noise gaussian --noise_mean 0 --noise_std 0.1`

**Salt and pepper noise:** `python code/algorithm/main.py --noise salt_and_pepper --noise_p 0.1 --noise_r 0.5`

**Missing block noise:** `python code/algorithm/main.py --noise missing_block --noise_blocksize 4 --noise_blocks 8`

You should only specify the noise parameters for the chosen type of noise, _e.g._ if using `uniform` noise, only specify `--noise_mean` and `--noise_std`, and not other parameters like `--noise_p`.

### Changing the model

To change the type of noise used, you can specify the `--model` parameter along with either `standard` for standard NMF (the default), `l1_robust` for RNMF or `l21` for $L_{2,1}$ NMF:

**Standard NMF**: `python code/algorithm/main.py --model standard`

**RNMF**: `python code/algorithm/main.py --model l1_robust`

**$L_{2,1}$ NMF**: `python code/algorithm/main.py --model l21`

### Additional configuration

To view additional command line arguments that are not mentioned here, you can use the help command as follows:

`python code/algorithm/main.py --help`

## Contributing

If you are making changes to the code base, make sure to install the `pre-commit` hook for the project to enable automatic code formatting and linting.
