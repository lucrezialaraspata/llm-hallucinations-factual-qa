# Probing Hallucinations through Knowledge Conflicts

This repository investigates how hallucinations relate to underlying knowledge conflicts. It provides tools and scripts to collect, analyze, and probe model outputs for factual inconsistencies, supporting research into model reliability and interpretability.

## ⚙️ Setup
Create and activate the uv virtual environment by running:
```bash
uv venv --python 3.11.5
source .venv/bin/activate
```
Set up the env by running `setup.sh`, which will install all the needed dependencies. 
To activate such venv, you can run the command `source .venv/bin/activate` in Linux environments.

## 📊 Data sources

An overview of the datasets/models used can be found in the paper under the section 4 **Experimental setup** section of the paper.
In particular, while our **result_collector.py** uses **TriviaQA** directly, for TREX we do/save a sampling in the form of founders/capitals/place_of_birth.csv.
Run `trex_parser.py` to create these data files.

## 📦 Artifact data collection

This project reproduces the original hallucination detection experiments using Falcon-7B and the TriviaQA dataset. Artifacts were collected by running `results_collector.py`, and probing models were trained and saved using `classifier_model.py`.

To extend the analysis to knowledge conflicts, the following scripts were added:
- `eval_datasets_nqswap.py`: Handles the NQ-Swap dataset for probing knowledge conflicts.
- `results_collector_kc.py`: Stores artifacts related to knowledge conflicts.
- `predict_kc_by_hall.py`: Uses the previously trained hallucination probing models to probe knowledge conflicts.

The workflow now supports both hallucination detection and knowledge conflict probing using the collected artifacts and trained models.

> [!CAUTION]
>
> Artifact data collection is done in **result_collector.py**, is **VERY** time consuming and best done on a powerful machine.
> It will write picke files and it gathers more data than used in the paper (in the paper we look at last layer activations, etc).
> Once acquired however, the same data can be used for a broader analysis if so desired.

We use models/tokenizers from Huggingface. Softmax/logits are collected directly from the model, attributions are collected using the 
integrated gradients (IG) method available in Captum and activations and attentions (model internal states) are collected using the **register_forward_hook** functionality.

## 📈 Plots

A graphical representation of probing models' performance across tasks, layers, and activation types is provided in the notebook **plot_accuracy.ipynb**.
