# PTBXL-ECG-Research

Research code and selected reproducibility materials for multi-label ECG abnormality classification, probability calibration, and explainability using the PTB-XL dataset.

## Research Scope

This repository covers three research components:

1. Multi-label ECG classification using Lightweight CNN1D and ResNet1D.
2. Probability calibration using global sigmoid temperature scaling.
3. Behavioral explanation assessment using Integrated Gradients (IG) and signal masking.

The study considers five PTB-XL diagnostic superclasses:
NORM, MI, STTC, CD, and HYP.

## Dataset

- Dataset: PTB-XL
- Version: 1.0.3
- Sampling rate used in the project: 100 Hz
- Data partitioning: Patient-disjoint training, validation, and test sets.

The original ECG recordings are not included in this repository.
Users must obtain PTB-XL from its official distribution source.

## Repository Contents

- `config/`: Dataset configuration.
- `scripts/`: Data preparation, dataset verification, split validation, and statistical analysis.
- `src/`: Dataset utilities, CNN1D, ResNet1D, and IG implementation.
- `logs/`: Selected training, validation, threshold selection, calibration, and explanation scripts.
- `requirements.txt`: Recorded Python package versions.

## Environment

- Python 3.12.10
- PyTorch 2.11.0+cu128
- Windows 11

Other recorded dependencies are listed in `requirements.txt`.

## Reproducibility Status

This repository contains selected source code from the research project.

Complete end-to-end reproduction of all reported results has not yet been independently verified. In particular, the full final-test evaluation and all statistical reporting procedures are not yet established as a verified public execution pipeline.

The results are limited to the experimental conditions defined in the study. They do not establish external clinical validity or readiness for clinical deployment.

## Citation

If using this repository, please cite the associated research work when its final bibliographic details become available.
