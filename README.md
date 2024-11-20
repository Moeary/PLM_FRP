# Ferroptosis-Related Protein Prediction (FPP)

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

```bash
git clone 
```


### Create virtual venv



To install the required packages, run:

```bash
pip install tensorflow==2.1.5 scikit-learn==0.24.2 xgboost==1.4.2 pandas==1.1.5 numpy==1.19.5
```

### Usage

1. Training the Model

Prepare your feature files in the specified format.

Run the training script with the desired features and classifier

2. LambdaRank

To run the LambdaRank model, execute:

```bash
python LambdaRank.py
```

3. Cross-Validation(Focus on )

To perform cross-validation, run:

```bash
python cross_vallidation.py
```