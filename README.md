# Voxelwise Encoding Models with Multimodal Features and fMRI Data

## Project Overview

This project focuses on building voxelwise encoding models using features extracted from large multimodal models (MMLs) and fMRI data. The goal is to use these features to predict brain activity and provide a comparative analysis between human brain responses and the internal representations of large MMLs. By doing so, this project aims to contribute to model interpretability, bridging the gap between artificial intelligence and cognitive neuroscience.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Data](#data)
- [Methodology](#methodology)
- [Results](#results)
- [Conclusion](#conclusion)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

In recent years, large multimodal models (MMLs) have demonstrated remarkable capabilities across various tasks. However, understanding how these models process information and how their internal representations compare to human brain activity remains an open challenge. This project seeks to address this challenge by developing voxelwise encoding models that map the relationship between features extracted from MMLs and brain activity measured via fMRI.

## Features

- **Voxelwise Encoding**: Models that predict brain activity on a voxel-by-voxel basis.
- **Multimodal Feature Extraction**: Utilizes features from state-of-the-art multimodal models.
- **Comparative Analysis**: Compares the internal representations of MMLs to human brain activity.
- **Interpretability**: Aims to provide insights into the interpretability of MMLs through the lens of cognitive neuroscience.

## Data

- **fMRI Data**: Functional Magnetic Resonance Imaging data representing brain activity.
- **MML Features**: Features extracted from large multimodal models, such as text, image, or combined modalities.

### Data Preprocessing

Details on how the data is preprocessed, including any normalization, scaling, or other transformations applied to the fMRI data and MML features.

## Methodology

### Model Architecture

- Description of the voxelwise encoding model used.
- Details on how MML features are mapped to fMRI voxels.

### Training

- Overview of the training process, including loss functions, optimization algorithms, and any regularization techniques applied.

### Evaluation

- Methods for evaluating model performance, such as correlation between predicted and actual brain activity.
- Explanation of how comparative analysis is conducted between human brain activity and MML internal representations.

## Results

- Summary of the key findings from the voxelwise encoding models.
- Visualizations comparing the predictions to actual brain activity.
- Insights into the similarities and differences between human brain activity and MML representations.

## Conclusion

- Discussion of the implications of the results for model interpretability and cognitive neuroscience.
- Potential applications of the findings.
- Future work and directions for further research.

## Installation

Instructions on how to install the necessary dependencies and set up the environment to run the project.

```bash
# Example command to set up the environment
pip install -r requirements.txt
```

## Acknowledgements
All researchers who've contributed to the literature on voxelwise encoding models and particularly using them 
with AI models. The papers below were crucial to the development of this project:

[Better models of human high-level visual cortex emerge from natural language supervision with a large and diverse dataset](https://www.nature.com/articles/s42256-023-00753-y)

[Brain encoding models based on multimodal
transformers can transfer across language and vision](https://arxiv.org/abs/2305.12248)