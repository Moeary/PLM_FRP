# Pretrain Language Model-based Ferroptosis-Related Protein Prediction (PLM-FRP)

### Introduction

**Ferroptosis**, an iron-dependent form of regulated cell death, plays a critical role in various diseases. Accurate identification of ferroptosis-related proteins (FRPs) is crucial for understanding its mechanisms and developing therapeutic strategies. Existing computational methods for FRP prediction often suffer from limited accuracy. In this study, we leverage the power of pre-trained protein language models to develop a novel deep learning model, named FPP, for FRP prediction. By integrating embeddings from ESM2, FPP captures complex relationships within protein sequences, achieving a remarkable 96.09% accuracy on a benchmark dataset, significantly outperforming previous methods and establishing a new state-of-the-art. Our findings suggest that FPP can serve as a powerful tool for FRP prediction and facilitate ferroptosis research. Our study leverages deep learning models to predict FRPs, which can facilitate a deeper understanding of ferroptosis mechanisms and provide guidance for the development of novel ferroptosis regulators.

### Requirements

```
Python 3.10
TensorFlow 2.1.5
scikit-learn 0.24.2
xgboost 1.4.2
pandas 1.1.5
numpy 1.19.5
Installation
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Moeary/PLM_FRP.git
```

2. Create and activate a virtual environment using conda:
```bash
conda create -n PLM_FRP
conda activate PLM_FRP
```

3. Install the required packages:

```bash
pip install tensorflow==2.1.5 scikit-learn==0.24.2 xgboost==1.4.2 pandas==1.1.5 numpy==1.19.5
```

### Usage

1. Training the Model

The `train.py` script allows you to train and evaluate different models with various feature combinations.

**Usage:**
```bash
python train.py --features <feature1>+<feature2> --classifier <classifier_name>
```

**Arguments:**
* `--features`: Select one or more features to use for training. Multiple features can be combined using `+`.
  * Available features: `aac`, `aadp`, `cksaap`, `dde`, `esm1b`, `esm2`, `gtpc`, `kbigram`, `sfpssm`, `protbert`.
* `--classifier`: Select the classifier to use.
  * Available classifiers: `cnn`, `svm`, `rf`, `nb`, `xgb`, `bilstm`.

**Example:**
```bash
python train.py --features dde+esm1b --classifier cnn
```
The evaluation results (Accuracy, AUC, Sensitivity, Specificity, MCC) will be printed to the console and appended to `evaluation_results.csv`.

2. LambdaRank

To run the LambdaRank model for feature selection, execute:

```bash
python LambdaRank.py
```

**Note:** The `LambdaRank.py` script is provided as an example of a feature selection method and its results do not represent the final performance of the PLM_FRP model.

3. Cross-Validation (Focus on 216D Selected Feature)

The cross-validation script focuses on evaluating the model using a selected set of 216-dimensional features. This process helps in assessing the model's performance and robustness. To perform cross-validation, run:

```bash
python cross_vallidation.py
```

### Significance

Ferroptosis, a recently discovered form of regulated cell death characterized by iron-dependent lipid peroxidation, has emerged as a critical process in various pathologies, including cancer, neurodegenerative disorders, and ischemia-reperfusion injury. Identifying and characterizing FRPs is paramount for unraveling the molecular mechanisms underlying ferroptosis, enabling the development of targeted therapeutic interventions, and ultimately improving patient outcomes. Traditional experimental methods for identifying and characterizing FRPs are time-consuming and resource-intensive. Computational methods, particularly those based on machine learning, offer a promising alternative for large-scale and efficient FRP prediction. Our model, FPP, leverages deep learning and pre-trained protein language models to achieve state-of-the-art performance in FRP prediction, significantly advancing the field of ferroptosis research.