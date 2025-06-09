# Hate‑Speech Detection in Memes – EE‑559 Final Project (Group 32)

**Authors:** Alix Benoit · Aaron Dinesh · Fabio Marcinno’
**Course:** EE‑559 — Deep Learning, EPFL (Spring 2025)

---

## Project Overview

Online memes often combine text and imagery, making hateful or abusive content difficult to detect with single‑modality models. In this project we:

* **Train** the lightweight **MultiModN (MMN)** architecture on the **MMHS150K** dataset.
* **Fine‑tune** the **LLaVA v1.6 Mistral 7B (LM‑7B)** multimodal LLM.
* **Evaluate** zero‑shot performance of **Llama‑4‑Scout‑17B‑16E‑Instruct (LSI‑17B)**.

Our codebase offers a **training & inference pipeline**, along with utilities for data wrangling, analysis and interactive labelling.

---

## Repository Structure

```text
├── app/               # Tools for quick graphing + manual labelling
├── llm_detection/     # Training & inference code for LM‑7B and LSI‑17B
│   ├── configs/       # Hydra/Lightning configs for reproducible runs
│   └── prompts/       # Prompt templates for zero‑shot inference
├── MultiModN/         # Fork of MMN with our MMHS150K pipeline additions
├── notebooks/         # Data exploration & result visualisation
├── env_requirements/  # Conda + pip requirements for each sub‑project
└── README.md          # You are here 
```

---

## Setup (Quick Start)

1. **Clone** the repo and submodules:

```bash
git clone --recursive <this‑repo‑url>
cd hate‑speech‑in‑memes
```

2. **Download MMHS150K** images & metadata and place them under `data/MMHS150K/`:

One can run the 'notebooks/mmhs_dataset_exploration.ipynb' notebook to download and unzip it automatically,
or one can download it directly from https://drive.usercontent.google.com/download?id=1S9mMhZFkntNnYdO-1dZXwF_8XIiFcmlF&export=download&authuser=0&confirm=t&uuid=db0e5b73-4ef4-45a4-b8f9-ef6f9c774473&at=APcmpozKaSM48fu1xNnp1-SNKDp1:1745766661322 .

3. Go to **README of desired directory**:

Each directory contains its own README with further instructions, for example to run the LLM pipeline, go to the README in the llm_detection directory.


## Results (Test Set)

| Classifier              | Hard Acc ↑ | Soft Acc ↑ | Cohen κ ↑ |    MAE ↓ |      F1 ↑ |   RMSE ↓ |
| ----------------------- | ---------: | ---------: | --------: | -------: | --------: | -------: |
| Random (imbal.)         |     27.0 % |     49.8 % |     0.001 |     1.10 |     0.597 |     1.39 |
| **LSI‑17B (zero‑shot)** |     22.2 % |     53.3 % |     0.027 |     1.31 |     0.659 |     1.63 |
| **LM‑7B (balanced)**    |     44.0 % |     69.2 % |     0.199 |     0.74 |     0.684 |     1.05 |
| **MMN (balanced)**      |     37.0 % |     64.4 % |     0.124 |     0.86 |     0.437 |     1.18 |
| Students (human)        |     32.5 % |     69.5 % |     0.048 |     0.93 |     0.301 |     1.24 |

> *Hard accuracy* checks exact score (0–3). *Soft accuracy* groups 0–1 = non‑hate, 2–3 = hate.

---

## Key Takeaways

* **Balanced data** significantly improves both MMN and LM‑7B.
* **LM‑7B** achieves the best overall metrics at one‑quarter the parameters of GPT‑4‑class models.
* **MMN** offers a strong trade‑off: >64 % soft accuracy with orders‑of‑magnitude fewer parameters and built‑in interpretability (sequential fusion).

See the full analysis, confusion matrices and hyper‑parameter ablations in the project report.

---

## Limitations & Future Work

* **Subjectivity** – annotator agreement for hatefulness is low (κ ≈ 0.35). Future work could model uncertainty explicitly.
* **OCR noise** – corrupted text degrades performance. 
* **Context missing** – replies often depend on tweet history.
* **Energy use** – even a 7 B LLM is costly.


---

## Acknowledgements

* Swamy *et al.* (2023) for **MultiModN**.
* Liu *et al.* (2023) for **LLaVA v1.6**.
* RCP AI‑as‑a‑Service for LSI‑17B access.
* Gomez *et al.* (2020) for **MMHS150K** dataset.


