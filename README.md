# Unsupervised Feature Matching for Affine Histological Image Registration

> [Unsupervised Feature Matching for Affine Histological Image Registration](https://link.springer.com/chapter/10.1007/978-3-031-78201-5_3)  
> Vladislav Pyatov and Dmitry Sorokin
> ICPR 2024

![Alt text](assets/method_overview.png?raw=true "Title")

## Training

The training can be reproduced with:
```shell
cd feature_
bash scripts/anhir_quadtree_ds.sh
```
* For dataset setup check the TissueDataset class


## Evaluation

Please follow instructions from the [ANHIR](https://anhir.grand-challenge.org) benchmark

## Acknowledgement

Our code is based on several awesome repositories:
- [LoFTR](https://github.com/zju3dv/LoFTR)
- [QuadTree Attention](https://github.com/Tangshitao/QuadTreeAttention)
- [DeepHistReg](https://github.com/MWod/DeepHistReg)

We thank the authors for releasing their code!

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@InProceedings{pyatov2024unfema,
author="Pyatov, Vladislav A. and Sorokin,Dmitry V.",
title="Unsupervised Feature Matching for Affine Histological Image Registration",
booktitle="Pattern Recognition. ICPR 2024",
year="2025",
publisher="Springer Nature Switzerland",
}
```