# Intermediate Machine Learning with scikit-learn
### Evaluation, Calibration, and Inspection

*By Thomas J. Fan*

[Link to slides](https://thomasjpfan.github.io/ml-workshop-intermediate-2-of-2/)

Scikit-learn is a machine learning library in Python that is used by many data science practitioners. In this training, we will learn about model evaluation, model calibration, and model inspection. For model evaluation, we will compare various metrics such as ROC AUC and mean average precision and see how they behave on datasets with different characteristics. We will use scikit-learn's plotting API to easily visualize the performance of a model and to compare multiple models. A well-calibrated model will predict probabilities that reflect the true likelihood of an event. Next, we will learn about techniques used for inspecting open-box machine learning models after they are trained. Afterwards, we will learn about inspection techniques used for more opaque models such as random forests or gradient boosted trees. These techniques are flexible because they can be applied to any machine learning model and gives a glimpse into how the model is generating its predictions.

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
