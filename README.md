# DUSpeckNet: Domain-Specific Augmentation and Uncertainty Quantification for Breast Ultrasound

Dual-task deep learning for breast ultrasound lesion **segmentation** and malignancy **classification**, combining ultrasound-specific data augmentation with Monte Carlo Dropout uncertainty. Includes **DUSpeckNet**, a lightweight (6.5M-parameter) dual-task network trained entirely from scratch for resource-constrained deployment.

This repository accompanies the paper *"Domain-Specific Augmentation and Uncertainty Quantification for Robust Breast Ultrasound Lesion Segmentation and Classification"* (MICCAI MIRASOL 2026).

> **Note on naming:** the model is referred to as `USNet` in parts of the code and as **DUSpeckNet** in the paper; they are the same architecture.

> ⚠️ **Release status:** For now, we are releasing only the **Kaggle notebook** containing the core implementation. The complete, organised pipeline (training, evaluation, and cross-domain experiment scripts, configs, and SLURM submission files) will be released **soon, immediately after the conference**. Please check back, or watch this repository for updates.

## Highlights

- **Dual-task framework:** a shared encoder feeds a U-Net-style segmentation decoder and a classification head, evaluated across ResNet34, ResNet50, and EfficientNet-B3 backbones.
- **Ultrasound-specific augmentation (US-Aug):** speckle noise, acoustic shadowing, and gain variation, modelling real B-mode artifacts. Improves segmentation and classification consistently across all backbones.
- **DUSpeckNet:** a from-scratch lightweight network (Speckle-Aware Stem, MBConv encoder, spatial-attention decoder, temperature-scaled head) that stays competitive at 51.8% fewer parameters, with no pretraining.
- **Uncertainty quantification:** Monte Carlo Dropout (N=20) provides per-image segmentation and classification uncertainty, with an analysis of its behaviour under domain shift.
- **External evaluation:** a preliminary, segmentation-only assessment on the ABreast cohort (community-acquired point-of-care ultrasound, Nigeria and Uganda).

## Results (5-fold CV)

| Configuration | Dice | IoU | Acc. | F1 | AUC |
|---|---|---|---|---|---|
| ResNet50 + US-Aug | **0.8814** | **0.7879** | 0.8860 | 0.8207 | 0.9396 |
| EfficientNet-B3 + US-Aug | 0.8774 | 0.7815 | **0.8936** | **0.8365** | **0.9432** |
| DUSpeckNet (ours, 6.5M) | 0.8539 | 0.7451 | 0.8615 | 0.7822 | 0.9155 |

On the external ABreast cohort (n=20, all benign), transfer is assessed for segmentation only, as the benign-only composition precludes evaluating malignancy detection. Zero-shot Dice: EfficientNet-B3 0.8485, DUSpeckNet 0.8190. Pretrained classification uncertainty tracked error (Spearman r=0.34, p<0.001) while the from-scratch model's did not (r=0.15, n.s.).

Full tables are in the paper.

## Repository structure

> The structure below reflects the **full pipeline to be released after the conference**. The initial release contains the Kaggle notebook only.

```
.
├── configs/            # experiment configuration files
├── src/
│   ├── data/           # dataset, dataloaders, augmentation, TTA
│   ├── models/         # DUSpeckNet / USNet and backbone variants
│   ├── training/       # training loop and losses
│   └── evaluation/     # metrics, MC Dropout uncertainty, correlation analysis
├── scripts/            # entry-point scripts
├── slurm/              # SLURM submission files
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/aondonamoses/duspecknet-breast-ultrasound.git
cd duspecknet-breast-ultrasound
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.10+, PyTorch 2.11, CUDA 12.8, on an NVIDIA Tesla P100 (16 GB).

## Data

The datasets are **not** included in this repository. Download each from its source and arrange it as described in [`data/README.md`](data/README.md):

- **BUSI** — breast ultrasound images (normal cases excluded).
- **BrEaST** — curated benchmark dataset.
- **BUS-BRA** — breast ultrasound dataset with patient identifiers.
- **ABreast** — African Breast Imaging Dataset for Equitable Cancer Care (external test cohort).

Each split is described by a `manifest.csv` with columns `case_id, image, mask, label, source`. The combined training set totals 2,773 images (1,859 benign, 914 malignant).

> **Cross-validation note:** splits are performed at the image level. Among the sources, only BUS-BRA contains multiple images per patient, so a subset of BUS-BRA patients may appear in both training and validation folds; BrEaST and BUSI are unaffected.

## Usage

> The scripts below describe the **full pipeline that will be released after the conference**. For now, the core implementation is available as a **Kaggle notebook**.

Train a configuration:

```bash
python scripts/train.py --config configs/effnetb3_usaug.yaml
```

Evaluate on the held-out folds:

```bash
python scripts/evaluate.py --config configs/effnetb3_usaug.yaml --checkpoint checkpoints/effnetb3_usaug_fold0.pth
```

Run the ABreast cross-domain experiments (zero-shot transfer, uncertainty under domain shift, leave-one-out fine-tuning):

```bash
python scripts/abreast_experiments.py --checkpoint checkpoints/usnet_fold0_best.pth
```

On a SLURM cluster, submit with:

```bash
sbatch slurm/train.sbatch
```

## Method summary

1. **Preprocessing.** Images resized to 256×256, ImageNet channel normalisation, grayscale replicated to three channels; masks binarised at threshold 127.
2. **Augmentation.** Standard (flip, rotation, scaling, brightness/contrast) plus three ultrasound-specific transforms: speckle noise, acoustic shadowing, and gain variation.
3. **Loss.** Weighted multi-task objective: `L = 0.6·L_seg + 0.4·L_cls`, with `L_seg` combining Dice and BCE equally.
4. **Uncertainty.** MC Dropout with N=20 stochastic forward passes; per-image classification uncertainty is the summed across-pass variance of class probabilities, and segmentation uncertainty the mean across-pass pixel variance.

## Citation

```bibtex
@inproceedings{iorumbur2026duspecknet,
  title     = {Domain-Specific Augmentation and Uncertainty Quantification for Robust Breast Ultrasound Lesion Segmentation and Classification},
  author    = {Iorumbur, Aondona Moses and Sanni, Henry Ananyi and Raymond, Confidence and Anazodo, Udunna},
  booktitle = {MICCAI MIRASOL Workshop},
  year      = {2026}
}
```

## Acknowledgements

The authors thank the clinicians at the Medical Artificial Intelligence Laboratory (MAI Lab) for their work in assembling the ABreast dataset and for making it available for this study, and gratefully acknowledge RISE-MICCAI for a travel grant supporting attendance and presentation at the conference.

## License

Released under the MIT License. See [`LICENSE`](LICENSE).