# Intermediate Machine Learning with scikit-learn
### Evaluation, Calibration, and Inspection

*By Thomas J. Fan*

[Link to slides](https://thomasjpfan.github.io/ml-workshop-intermediate-2-of-2/)

Scikit-learn is a Python machine learning library used by data science practitioners from many disciplines. We will learn about evaluating, calibrating, and inspecting models during this training. Model evaluation is an essential piece of the ML workflow. We will cover multiple metrics and see how they behave on various combinations of datasets and models. We will explore scikit-learn's plotting API to visualize a model's performance. Next, we will learn how to calibrate a machine learning model with scikit-learn. We will see how models behave before and after calibrating by visualizing an estimator's calibration. Next, we will explore techniques to inspect machine learning models. Specifically, we will see how to examine open-box machine learning models, such as linear models and random forests. Finally, we will learn about inspection techniques that apply to all models. These techniques are flexible because they can be used in any machine learning model and show how it generates predictions.

## Obtaining the Material

### With git

The most convenient way to download the material is with git:

```bash
git clone https://github.com/thomasjpfan/ml-workshop-intermediate-2-of-2
```

Please note that I may add and improve the material until shortly before the session. You can update your copy by running:

```bash
git pull origin master
```

### Download zip

If you are not familiar with git, you can download this repository as a zip file at: [github.com/thomasjpfan/ml-workshop-intermediate-2-of-2/archive/master.zip](https://github.com/thomasjpfan/ml-workshop-intermediate-2-of-2/archive/master.zip). Please note that I may add and improve the material until shortly before the session. To update your copy please re-download the material a day before the session.

## Running the notebooks

### Local Installation

Local installation requires `conda` to be installed on your machine. The simplest way to install `conda` is to install `miniconda` by using an installer for your operating system provided at [docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html). After `conda` is installed, navigate to this repository on your local machine:

```bash
cd ml-workshop-intermediate-2-of-2
```

Then download and install the dependencies:

```bash
conda env create -f environment.yml
```

This will create a virtual environment named `ml-workshop-intermediate-2-of-2`. To activate this environment:

```bash
conda activate ml-workshop-intermediate-2-of-2
```

Finally, to start `jupyterlab` run:

```bash
jupyter lab
```

This should open a browser window with the `jupterlab` interface.

### Run with Google's Colab

If you have any issues with installing `conda` or running `jupyter` on your local computer, then you can run the notebooks on Google's Colab:

1. [Model Evaluation](https://colab.research.google.com/github/thomasjpfan/ml-workshop-intermediate-2-of-2/blob/master/notebooks/01-model-evaluation.ipynb)
2. [Model Calibration](https://colab.research.google.com/github/thomasjpfan/ml-workshop-intermediate-2-of-2/blob/master/notebooks/02-model-calibration.ipynb)
3. [Model Inspection](https://colab.research.google.com/github/thomasjpfan/ml-workshop-intermediate-2-of-2/blob/master/notebooks/03-model-inspection.ipynb)

## License

This repo is under the [MIT License](LICENSE).
