## CQ-CNN: A Hybrid Classical-Quantum Convolutional Neural Network

<p align="left">
<img src="plots/github/qcnn-arch.jpg" alt="image_description" style="padding: 5px" width="100%">
</p>

<b>CQ-CNN: A Hybrid Classical-Quantum Convolutional Neural Network for Alzheimer’s Disease Detection Using Diffusion-Generated and U-Net Segmented 3D MRI</b>

<p>Mominul Islam, Mohammad Junayed Hasan, M.R.C. Mahdy</p>

[https://doi.org/10.48550/arXiv.2503.02345](https://doi.org/10.48550/arXiv.2503.02345)<br>

<p>Abstract: <i>The detection of Alzheimer’s disease (AD) from clinical MRI data is an active area of research in medical imaging Recent advances in quantum computing, particularly the integration of parameterized quantum circuits (PQCs) with classical machine learning architectures, offer new opportunities to develop models that may outperform traditional methods. However, quantum machine learning (QML) remains in its early stages and requires further experimental analysis to better understand its behavior and limitations. In this paper, we propose an end-to-end hybrid classical-quantum convolutional neural network (CQ-CNN) for AD detection using clinically formatted 3D MRI data. Our approach involves developing a framework to make 3D MRI data usable for machine learning, designing and training a brain tissue segmentation model (SkullNet), and training a diffusion model to generate synthetic images for the minority class. Our converged models exhibit potential quantum advantages, achieving higher accuracy in fewer epochs than classical models. The proposed 𝛽<sub>8</sub>-3-qubit model achieves an accuracy of 97.50%, surpassing state-of-the-art (SOTA) models while requiring significantly fewer computational resources. In particular, the architecture employs only 13K parameters (0.48 MB), reducing the parameter count by more than 99.99% compared to current SOTA models. Furthermore, the diffusion-generated data used to train our quantum models, in conjunction with real samples, preserve clinical structural standards, representing a notable first in the field of QML. We conclude that CQ-CNN architecture-like models, with further improvements in gradient optimization techniques, could become a viable option and even a potential alternative to classical models for AD detection, especially in data-limited and resource-constrained clinical settings.</i></p>

## 1. Specification of dependencies
This code requires two separate conda environments. Run the following to install the required packages on Windows.

```python
cd environments/
conda env create -f quantum.yml
conda env create -f classical.yml
```
The files inside the following paths require the `quantum` environment to be activated:

```
scripts/
|-- classifiers/
|---- 2qubits/
|------ ...
|---- 3qubits/
|------ ...
```

Run the following to activate the `quantum` environment.

```python
conda activate quantum
```

The rest of the notebooks require the `classical` environment to be activated. Run the following to activate the `classical` environment.

```python
conda activate classical
```
## 2a. Datasets
Download the [NFBS](http://preprocessed-connectomes-project.org/NFB_skullstripped) and [OASIS-2](https://sites.wustl.edu/oasisbrains/home/oasis-2) datasets. Once downloaded, create a folder named `datasets` in the root directory. Organize the folders, downloaded `.tar.gz` files, and metadata in the following structure:

```
datasets/
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

Note that you may need to rename the file and folder names as required.

## 2b. Pre-processing
To begin the pre-processing, run the following scripts associated with each dataset:

```
nfbs-preprocessing.ipynb
```
```
oasis-2-preprocessing.ipynb
```

## 3. Trained Models

We provide pre-trained models for segmentation and generative tasks.

### Segmentation Model
You can download the segmentation model from the [GitHub Releases v1.0.1](https://github.com/mominul-ssv/alz-cq-cnn/releases/tag/v1.0.1).

### Generative Models
You can download the generative models from the [GitHub Releases v1.0.0](https://github.com/mominul-ssv/alz-cq-cnn/releases/tag/v1.0.0).

| Model Type | Download Link |
|------------|---------------|
| **Axial**  | [Download](https://github.com/mominul-ssv/alz-cq-cnn/releases/download/v1.0.0/oasis-2_axial_gen_model.pt) |
| **Coronal**| [Download](https://github.com/mominul-ssv/alz-cq-cnn/releases/download/v1.0.0/oasis-2_coronal_gen_model.pt) |
| **Sagittal**| [Download](https://github.com/mominul-ssv/alz-cq-cnn/releases/download/v1.0.0/oasis-2_sagittal_gen_model.pt) |

## 5. Citation
```
@article{islam2025cq,
  title={CQ CNN: A Hybrid Classical Quantum Convolutional Neural Network for Alzheimer's Disease Detection Using Diffusion Generated and U Net Segmented 3D MRI},
  author={Islam, Mominul and Hasan, Mohammad Junayed and Mahdy, MRC},
  journal={arXiv preprint arXiv:2503.02345},
  year={2025}
}
```

## 6. License
- Copyright © Mominul Islam.
- ORCID iD: https://orcid.org/0009-0001-6409-964X


