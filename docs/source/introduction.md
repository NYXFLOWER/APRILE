
# Introduction

**Adverse Polypharmacy Reaction Intelligent Learner and Explainer (APRILE)** is an explainable framework to reveal the mechanisms underlying adverse drug reactions (ADRs) caused by polypharmacy therapy. After learning from massive biomedical data, APRILE generate a small pharmacogenomic knowledge graph (i.e. drug targets and protein interactions) as mechanistic explanation for a drug-drug interaction (DDI) which associated an ADR and a set of such interactions.

## Features
APRILE has three key features:

- Predicts side effects for drug combinations and gives the prediction reasons
- Delineate non-intuitive mechanistic associations between {genes, proteins, biological processes} and {symptoms, diseases, mental disorders ∈ ADRs}
- Explore molecular mechanisms for 843,318 (learned) + 93,966 (novel) side effect–drug pair events, spanning 861 side effects (472 diseases, 485 symptoms and 9 mental disorders) and 20 disease categories, have been suggested.

APRILE is able to answer the following example questions:
- Why the combination use of a pair of drugs (nicotine, ondansetron) causes anxiety?
- When taking fexofenadine, hydroxyzineand and loratadine simultaneously, what side effects may occur, and why?
- Which genes are associated with the infection diseases?
- What are the common mechanisms among peptic ulcers (such as duodenal ulcer, gastric ulcer and esophageal ulcer)?

We have demonstrated the viability of discovering polypharmacy side effect mechanisms by learning from an AI model trained on massive biomedical data (see our [[paper]](https://www.biorxiv.org/content/10.1101/2021.07.02.450937v1))

<!-- How to use `APRILE` answering such example questions are available at here: [[Jupyter notebook tutorials]]() -->