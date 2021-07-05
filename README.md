<!-- <p align="center"><img src="https://github.com/NYXFLOWER/APRILE/raw/main/docs/images/aprile_logo_long.png" alt="logo" width="600px" /></p> -->

# APRILE
<!-- ----------------------------------------- -->

[![PyPI version](https://img.shields.io/pypi/v/aprile?color=blue)](https://pypi.org/project/aprile/)
[![Documentation Status](https://readthedocs.org/projects/pykale/badge/?version=latest)](https://aprile.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/github/license/nyxflower/aprile?style=plastic)](https://github.com/NYXFLOWER/PoSe-Path/blob/master/LICENSE)
![Last-Commit](https://img.shields.io/github/last-commit/nyxflower/aprile?style=plastic)
![Size](https://img.shields.io/github/repo-size/nyxflower/aprile?color=green&style=plastic)


[**Getting Started**](https://github.com/NYXFLOWER/APRILE#installation) |
[**Documentation**](https://aprile.readthedocs.io/) |

**Adverse Polypharmacy Reaction Intelligent Learner and Explainer (APRILE)** is an explainable framework to reveal the mechanisms underlying adverse drug reactions (ADRs) caused by polypharmacy therapy. After learning from massive biomedical data, APRILE generate a small pharmacogenomic knowledge graph (i.e. drug targets and protein interactions) as mechanistic explanation for a drug-drug interaction (DDI) which associated an ADR and a set of such interactions.

APRILE is able to answer the following example questions:
- Why the combination use of a pair of drugs (nicotine, ondansetron) causes anxiety?
- When taking fexofenadine, hydroxyzineand and loratadine simultaneously, what side effects may occur, and why?
- Which genes are associated with the infection diseases?
- What are the common mechanisms among peptic ulcers (such as duodenal ulcer, gastric ulcer and esophageal ulcer)?

We have demonstrated the viability of discovering polypharmacy side effect mechanisms by learning from an AI model trained on massive biomedical data (see [[paper]](https://www.biorxiv.org/content/10.1101/2021.07.02.450937v1))

<!-- How to use `APRILE` answering such example questions are available at here: [[Jupyter notebook tutorials]]() -->

## Features
- APRILE predicts side effects for drug combinations and gives the prediction reasons
- APRILE delineates non-intuitive mechanistic associations between {genes, proteins, biological processes} and {symptoms, diseases, mental disorders $\in$ ADRs)
- Using our pre-trained model, molecular mechanisms for 843,318 (learned) + 93,966 (novel) side effect–drug pair events, spanning 861 side effects (472 diseases, 485 symptoms and 9 mental disorders) and 20 disease categories, have been suggested.

## Installation

***Prerequisites***:
Before installing `aprile`, [PyTorch](https://pytorch.org/) and [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric#installation) are required to be installed matching your hardware. 

*We recommend using torch 1.4.0 (python3.7+cuda10.1), torch-cluster 1.5.4, torch-scatter 2.0.4, torch-sparse 0.6.1, torch-spline-cov 1.2.0 and torch-geometric 1.4.2*

Install the environment dependencies of APRILE using `pip`:
```bash
pip install aprile
```

## Cite Us
If you found this work useful, please cite us:
```
@article{aprile,
	title={APRILE: Exploring the Molecular Mechanisms of Drug Side Effects with Explainable Graph Neural Networks},
	author={Hao Xu and Shengqi Sang and Herbert Yao and Alexandra I. Herghelegiu and Haiping Lu and Laurence Yang},
	journal={bioRxiv preprint},
	year={2021}
}
```
