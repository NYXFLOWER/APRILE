<!-- <p align="center"><img src="https://github.com/NYXFLOWER/APRILE/raw/main/docs/images/aprile_logo_long.png" alt="logo" width="600px" /></p> -->

# APRILE
<!-- ----------------------------------------- -->

[![PyPI version](https://img.shields.io/pypi/v/aprile?color=blue)](https://pypi.org/project/aprile/)
[![Downloads](https://pepy.tech/badge/aprile/month?style=plastic)](https://pepy.tech/project/aprile)
[![Downloads](https://pepy.tech/badge/aprile?style=plastic)](https://pepy.tech/project/aprile)
[![Documentation Status](https://readthedocs.org/projects/pykale/badge/?version=latest)](https://aprile.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/github/license/nyxflower/aprile?style=plastic)](https://github.com/NYXFLOWER/PoSe-Path/blob/master/LICENSE)
![Last-Commit](https://img.shields.io/github/last-commit/nyxflower/aprile?style=plastic)
<!-- ![Size](https://img.shields.io/github/repo-size/nyxflower/aprile?color=green&style=plastic) -->

[**Getting Started**](https://github.com/NYXFLOWER/APRILE#installation) |
[**Documentation**](https://aprile.readthedocs.io/) 
|
[**Package**](https://pypi.org/project/aprile/)
|
[**Paper**](https://www.biorxiv.org/content/10.1101/2021.07.02.450937v1)

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
- APRILE delineates non-intuitive mechanistic associations between {genes, proteins, biological processes} and {symptoms, diseases, mental disorders ∈ ADRs)
- Using our pre-trained model, molecular mechanisms for 843,318 (learned) + 93,966 (novel) side effect–drug pair events, spanning 861 side effects (472 diseases, 485 symptoms and 9 mental disorders) and 20 disease categories, have been suggested.

## Installation

***Prerequisites***:
Before installing `aprile`, [PyTorch](https://pytorch.org/) and [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric#installation) are required to be installed matching your hardware. 

*We recommend using torch 1.4.0 (python3.7+cuda10.1), torch-cluster 1.5.4, torch-scatter 2.0.4, torch-sparse 0.6.1, torch-spline-cov 1.2.0 and torch-geometric 1.4.2*

Install the environment dependencies of APRILE using `pip`:
```bash
pip install aprile
```

## Usage

Firstly, download the data file `kgdata.pkl` using this [link](https://drive.google.com/file/d/1ZT9VhybmnOxHsvzFvt7DKRQd3EZZheK9/view?usp=sharing), and put it into your working directory.

Secondly, load data and APRILE model.
```python
from aprile.model import *

gdata = AprileQuery.load_from_pkl('kgdata.pkl')	
aprile = Aprile(gdata, device='cuda')	          # device='cpu' if using CPUs
```
*\* If you prefer to use torch > 1.8 and torch-geometric > 2.0, see [#2](/../../issues/2) when you prepare data for aprile models.*

Next, let us familiar with the data `gdata`. It's data type is [`torch_geometric.data.data.Data`]((https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html)) and its attribute list can be obtained by using `var(gdata).keys()`. It mainly contains four parts: 
1) a pharmacogenomic knowledge graph: `gdata.pp_index` and `gdata.pd_index`
2) the ADRs caused by polypharmacy: `gdata.dd_edge_index`
3) the data for training and testing APRILE-Pred: `gdata.train_idx`, `gdata.train_et`, `gdata.test_idx` and `gdata.test_et`
4) the index mappings for drugs, genes, proteins and ADRs:
	- `gdata.side_side_effect_idx_to_name`: mapping from side effect aprile index to side effect name
	- `gdata.drug_idx_to_id`: mapping from drug aprile index to CID
	- `gdata.prot_idx_to_id`: mapping from protein aprile index to GeneID
	- `gdata.geneid2symbol`: mapping from GeneID to gene symbol

Finally, use APRILE to predict ADRs caused by polypharmacy and generate explanations (e.g. molecular mechanisms of the ADRs). Here is an example,
```python
# a list of DDIs in the formate of (D1, D2, SE)
d1, d2, se = [19, 37, 192], [37, 192, 19], [452]*3

# get predictions
query = aprile.predict(d1, d2, se)
# get prediction result table
pred_df = query.get_pred_table()

# get explain --> proteins and GOs
query = aprile.explain_query(query, regularization=2, if_auto_tuning=True)

# save query to file
query.to_pickle('tmp.pkl')

# load query from file
query = PoseQuery.load_from_pkl('tmp.pkl')

# print query summary
print(query)

# get detailed prediction and explaination results
prediction_df = query.get_pred_table()
go_df = query.get_GOEnrich_table()

# visualize explained query and save
subgraph_fig = query.get_subgraph(if_show=True, save_path='test.pdf')
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
