## CQ-CNN: A Lightweight Hybrid Classical–Quantum Neural Network

<p align="left">
<img src="plots/github/qcnn-arch.jpg" alt="image_description" style="padding: 5px" width="100%">
</p>

<b>CQ-CNN: A lightweight hybrid classical–quantum convolutional neural network for Alzheimer’s disease detection using 3D structural brain MRI</b>

<p>Mominul Islam, Mohammad Junayed Hasan, M.R.C. Mahdy</p>

[https://doi.org/10.1371/journal.pone.0331870](https://doi.org/10.1371/journal.pone.0331870)<br>

<p>Abstract: <i> The automatic detection of Alzheimer’s disease (AD) using 3D volumetric MRI data is a complex, multi-domain challenge that has traditionally been addressed by training classical convolutional neural networks (CNNs). With the rise of quantum computing and its potential to replace classical systems in the future, there is a growing need to: (i) develop automated systems for AD detection that run on quantum computers, (ii) explore the capabilities of current-generation classical-quantum architectures, and (iii) identify their potential limitations and advantages. To reduce the complexity of multi-domain expertise while addressing the emerging demands of quantum-based automated systems, our contribution in this paper is twofold. First, we introduce a simple preprocessing framework that converts 3D MRI volumetric data into 2D slices. Second, we propose CQ-CNN, a parameterized quantum circuit (PQC)-based lightweight hybrid classical-quantum convolutional neural network that leverages the computational capabilities of both classical and quantum systems. Our experiments on the OASIS-2 dataset reveal a significant limitation in current hybrid classical-quantum architectures, as they face difficulties con verging when class images are highly similar, such as between moderate dementia and non-dementia classes of AD, which leads to gradient failure and optimization stagnation. However, when convergence is achieved, the quantum model demonstrates a promising quantum advantage by attaining state-of-the-art accuracy with far fewer parameters than classical models. For instance, our 𝛽<sub>8</sub>-3-qubit model achieves 97.5% accuracy using only 13.7K parameters (0.05 MB), which is 5.67% higher than a classical model with the same parameter count. Nevertheless, our results highlight the need for improved quantum optimization methods to support the practical deployment of hybrid classical-quantum models in AD detection and related medical imaging tasks.</i></p>

## 1a. Dependency Setup

This project requires two separate conda environments. Follow these steps to install the necessary packages on Windows:

1. Navigate to the `environments/` directory:
    ```bash
    cd environments/
    ```

2. Create the conda environments using the provided YAML files:
    ```bash
    conda env create -f quantum.yml
    conda env create -f classical.yml
    ```

## 1b. Activating Environments

- **Quantum Environment:** The following files require the `quantum` environment to be activated:

    ```
    scripts/
    |-- classifiers/
    |---- 2qubits/
    |------ ...
    |---- 3qubits/
    |------ ...
    ```

    To activate the `quantum` environment, run:
    ```bash
    conda activate quantum
    ```

- **Classical Environment:** The remaining notebooks require the `classical` environment. To activate it, run:
    ```bash
    conda activate classical
    ```

## 2a. Datasets

Download the following datasets:

- [NFBS](http://preprocessed-connectomes-project.org/NFB_skullstripped)
- [OASIS-2](https://sites.wustl.edu/oasisbrains/home/oasis-2)

## 2b. Organizing Datasets

After downloading the datasets, create a folder named `datasets` in the root directory. Organize the folders, downloaded `.tar.gz` files, and metadata as shown below:

```
datasets/
|-- NFBS/
|---- downloads/
|------ NFBS_Dataset.tar.gz
|-- OASIS-2/
|---- downloads/
|------ OAS2_RAW_PART1.tar.gz
|------ OAS2_RAW_PART2.tar.gz
|------ OAS2_metadata.xlsx
```

Make sure to rename the files and folders as needed.

## 2c. Preprocessing NFBS and Training the Segmentation Model

To train the segmentation model, start by preprocessing the NFBS dataset. This process converts the 3D MRI data into 2D images. Run the following script for preprocessing:

```
nfbs-preprocessing.ipynb
```

After preprocessing, proceed to train the segmentation model with the following script:

```
nfbs-unet-train.ipynb
```

## 2d. Preprocessing OASIS-2, Training Diffusion Models, and Creating Dataset Variations

For the **OASIS-2** dataset, start by converting the 3D MRI data into 2D images. The images are then divided into classes based on metadata, and we create variations for axial, coronal, and sagittal views.

Run the preprocessing script:

```
oasis-2-preprocessing.ipynb
```

To address class imbalance, we train **three separate diffusion models** for the minority class (moderate dementia) in axial, coronal, and sagittal views. These models generate synthetic images. Run the following scripts to train the diffusion models:

```
oasis-2-diffusion-train-axial.ipynb
oasis-2-diffusion-train-coronal.ipynb
oasis-2-diffusion-train-sagittal.ipynb
```

These models are trained on [Kaggle](https://www.kaggle.com/). Once training is complete, create the following folder structure and store the generated images in the respective `moderate_dementia` folders:

```
datasets/
|-- OASIS-2/
|---- generated/
|------ axial/
|-------- moderate_dementia/
|------ coronal/
|-------- moderate_dementia/
|------ sagittal/
|-------- moderate_dementia/
```

Next, run the following script to create the final balanced variations of the dataset:

```
oasis-2-eda-and-pruning.ipynb
```

Finally, use the segmentation model to generate variations of the OASIS-2 dataset by running:

```
oasis-2-skullstrip.ipynb
```

## 2e. Trained Models

We provide pre-trained models for both segmentation and generative tasks.

### Segmentation Model
Download the segmentation model from [GitHub Releases v1.0.1](https://github.com/mominul-ssv/alz-cq-cnn/releases/tag/v1.0.1).

### Generative Models
Download the generative models from [GitHub Releases v1.0.0](https://github.com/mominul-ssv/alz-cq-cnn/releases/tag/v1.0.0).

| Model Type | Download Link |
|------------|---------------|
| **Axial**  | [Download](https://github.com/mominul-ssv/alz-cq-cnn/releases/download/v1.0.0/oasis-2_axial_gen_model.pt) |
| **Coronal**| [Download](https://github.com/mominul-ssv/alz-cq-cnn/releases/download/v1.0.0/oasis-2_coronal_gen_model.pt) |
| **Sagittal**| [Download](https://github.com/mominul-ssv/alz-cq-cnn/releases/download/v1.0.0/oasis-2_sagittal_gen_model.pt) |

## 3. Training Classifiers with CQ-CNNs

To train the classification models for all variations of the OASIS-2 dataset, run the scripts in the following directories:

```
scripts/
|-- classifiers/
|---- 2qubits/
|------ ...
|---- 3qubits/
|------ ...
|---- classical/
|------ ...
|---- classical-sota/
|------ ...
|---- control/
|------ ...
```

## 4. Performance Evaluation

For an in-depth analysis of our experiments, run the following scripts:

```
model-segmentation-eda.ipynb
model-generative-eda.ipynb
model-classification-eda.ipynb
model-classification-control-eda.ipynb
```

## 5. Citation
```
@article{islam2025cq,
  title={CQ-CNN: A lightweight hybrid classical--quantum convolutional neural network for Alzheimer’s disease detection using 3D structural brain MRI},
  author={Islam, Mominul and Hasan, Mohammad Junayed and Mahdy, MRC},
  journal={PloS one},
  volume={20},
  number={9},
  pages={e0331870},
  year={2025},
  publisher={Public Library of Science San Francisco, CA USA}
}
```

## 6. License
- Copyright © Mominul Islam.
- ORCID iD: https://orcid.org/0009-0001-6409-964X


