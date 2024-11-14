## Experiment materials for the paper "_CHyMER: A <ins>C</ins>ontext-based <ins>Hy</ins>brid Approach to <ins>M</ins>ining <ins>E</ins>thical Concern-related App <ins>R</ins>eviews_"

### Introduction

This repository includes all the materials (code, input data, evaluation results, output dataset) of our approach to extract ethical concern-related app reviews. Those materials involve three components of our framework: _NLI inference and evaluation, LLM inference and evaluation, and CHyMER (NLI+LLM)_.

### File Descriptions

There are three directories in total, and the content of each directory is described as follows:

`datafiles`: the input data used for this study.

- `privacy_gt_dataset.csv` is the ground truth dataset containing privacy or not-privacy labeled app reviews from mental health domain.
- `pseudo_labeled_corpus.parquet` is the NLI annotated dataset created using the best-performing NLI model and the best set of hypotheses in the NLI inference part.
- `MH_12star_reviews.csv` is the unlabeled dataset containing 1 and 2 star rated app reviews from mental health domain. We use this dataset extract new privacy related reviews.
- `chymer_classified_reviews.csv` is the dataset extracted using our proposed CHyMER approach.

`docs`: additional supporting documents for our study.

- `evaluation_results.md` contains the results of our NLI and LLM evaluation parts.
- `manual_label_guide.md` contains the instructions followed by the annotators for manual inspection.

`programfiles`: the source code of our study.

- `nli_inference.py` contains the code for NLI inference and evaluation (RQ1).
- `llm_inference.py` contains the code for LLM inference and evaluation (RQ2).
- `chymer.py` contains the code to extract new privacy reviews from the unlabeled dataset using the proposed NLI+LLM approach.
- `data_preprocess.py` contains the data pre-processing code used to clean the data before extracting privacy reviews.

`new_privacy_reviews.csv` is the manually curated dataset after using CHyMER on a set of 42,271 app reviews. 
