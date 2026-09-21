# Technical Report — Kuza AI: Offline Agricultural AI for East African Farmers

**Team ID:** kuza-ai
**Domain:** agriculture
**Model:** kuza-ud-q4_k_xl

---

## Problem

Kuza AI is an offline agricultural AI assistant for smallholder farmers and agricultural extension workers in East Africa. It is designed to provide practical agricultural guidance when extension services are capacity-constrained and reliable internet access is unavailable or expensive.

The target use cases include crop and livestock management, agricultural best practices, pest and disease guidance, and other farm-level advisory questions. The motivation is particularly strong in regions where extension-worker coverage is limited. For example, published reports from Uganda have described approximately one agricultural extension worker for every 5,000 farmers.

Kuza currently supports **English and Swahili** agricultural interactions, although Swahili factuality and response quality remain areas for further evaluation. It is also designed to run locally through `llama.cpp`, without external APIs or network access during inference. This makes the system suitable for low-connectivity environments and avoids requiring farmers to continuously send potentially sensitive agricultural queries to cloud services.

The model is an **experimental prototype**, not a replacement for qualified agricultural extension workers, veterinarians, or other agricultural professionals. In particular, safety-critical recommendations such as pesticide, chemical, or veterinary treatment decisions require further expert validation.

---



## Design Decisions



### Base model

I evaluated several candidate models using the ADTC profiler before selecting the final base model. All candidates were evaluated on the same machine, in the same environment and run configuration, using Unsloth's UD XL GGUF quantization approach.


| Candidate                   | Generation speed |
| --------------------------- | ---------------- |
| Gemma 4 E4B IT              | 2.24 tokens/s    |
| DeepSeek-R1-Distill-Qwen-7B | 1.46 tokens/s    |
| Gemma 4 E2B IT              | 4.70 tokens/s    |
| Phi-4                       | 3.60 tokens/s    |


Gemma 4 E2B provided the best throughput among the evaluated candidates while maintaining a relatively small model footprint. I therefore started from the Unsloth `gemma-4-E2B-it-qat-q4_0-unquantized` checkpoint. The checkpoint is QAT-oriented and provides a multimodal foundation for possible future image and audio agricultural use cases.

### Fine-tuning data

The primary objective of the data pipeline was to maximize **useful agricultural information per training example**, rather than simply maximizing dataset size. This was important because the final model needed to remain small enough for CPU-only, offline laptop inference.

The pipeline started from the Digital Green FarmerChat large dataset containing approximately **1.45 million farmer-initiated Q&A pairs**, including substantial African data such as Kenya and Ethiopia.

The preparation pipeline was:

1. Analyze the FarmerChat corpus and distinguish high-priority/frequent questions from long-tail questions, while using semantic representations to group related queries and improve sampling diversity.
2. Sample **50K priority examples and 20K long-tail examples** for further processing.
3. Use **Llama-3.3-70B-Versatile** to clean the questions and answers, remove filler, and increase information density.
4. Produce an approximately **56K-example intermediate corpus**.
5. From this corpus, remove malformed examples, duplicates, and examples whose answers were too semantically similar to other answers. The resulting final English training set contains exactly **25,506 examples** (`kuzaai/kuza_sft_english`).
6. Generate a corresponding Swahili SFT dataset from the English corpus using **GPT-OSS-120B (Groq)**, then apply numeral, glossary, and length QC. This results in **23,885 Swahili rows** (`kuzaai/kuza_sft_swahili`).
7. Add a small, hand-authored adversarial dataset (**130 rows**, split evenly between English and Swahili) for safety-oriented training. Prompts elicit unlabeled pesticide doses or illegal agrochemical requests, and assistant turns are trained to refuse and redirect to local extension officers (`kuzaai/kuza_sft_adversarial`).
8. Add a hand-authored multi-turn dataset (**104 rows**) where the assistant first asks clarifying questions (crop, location, symptom) before giving concrete advice (`kuzaai/kuza_sft_multiturn`).
9. Generate DPO preference pairs (**1,224 rows**: 1.1k train, 122 val) to align the model's helpfulness and refusal behaviors (`kuzaai/kuza_dpo_preference`).

The removed examples remain available in the intermediate `agri_sft_prod_56k` dataset but are not used for training the submitted model.

Large teacher models were used only during offline dataset preparation. The final submitted GGUF does not require these models or any external API during inference.

### Training Mix and Hyperparameters

To maintain broad instruction-following capabilities while specializing in agriculture, the final SFT training mixture included:

- 100% of `kuza_sft_english`
- 50% of `kuza_sft_swahili`
- 8% `HuggingFaceH4/no_robots` (general instruction)
- 5% `kuza_sft_adversarial`
- 3% `kuza_sft_multiturn`

**Supervised Fine-Tuning (SFT):**

- **Method:** QLoRA via Unsloth QAT int4
- **LoRA Config:** RsLoRA with rank `r=32`, `alpha=64`
- **Sequence Length:** 2048 tokens
- **Epochs:** 2
- **Learning Rate:** `1e-4`

**Direct Preference Optimization (DPO):**

- **Dataset:** 866 train / 97 validation pairs from `kuza_dpo_preference`
- **Parameters:** Beta `0.1`, RPO alpha `1.0`
- **Epochs:** 1
- **Learning Rate:** `5e-6`
- **Loss:** Sigmoid

The training stack was built on **Unsloth (2026.8.19)**, **Transformers (5.5.0)**, **TRL (0.24.0)**, **PEFT (0.19.1)**, and **PyTorch (2.10.0+cu128)**.

### Alternatives considered

I first explored constructing a dataset directly from approximately 200 agricultural extension guidelines, handbooks, and other public sources by extracting facts and reverse-generating conversations. This produced less useful data because the available material was difficult to gather systematically, was not sufficiently dense at the required scale, and generated questions were less representative of real farmer interactions than farmer-initiated queries.

I also investigated the two-stage agricultural advisory approach described in *Fine-Tuning and Evaluating Conversational AI for Agricultural Advisory*, where factual knowledge handling and conversational response generation are separated. This is a promising direction for a production system, but the ADTC submission requires a single GGUF model running through `llama.cpp`, so that architecture was not used in the submitted system.

### Quantization and Screening

The submitted artifact is `kuza-ud-q4_k_xl.gguf`: a **mixed-precision GGUF** built with an **importance matrix (**`imatrix`**)** and llama.cpp tensor overrides (bulk FFN in **Q4_K**, attention and selected embeddings/projections in **Q5_K**). The internal pipeline recipe name is `ud_q4_k_xl`; full tensor counts and overrides are in `[provenance/ud_q4_k_xl_recipe.json](provenance/ud_q4_k_xl_recipe.json)`.

Several quant candidates were built from the same merged Kuza reference weights and screened with KL divergence, hidden-prompt behavior, CPU throughput, size, and peak RSS. 

The screening pipeline (`06_screen.py`) applies quality gates (safety, language consistency, KLD ceiling) first. Control quants (like the 8-bit `q8_0_ceiling`) are disqualified from winning to ensure edge deployment viability. Among the gate-passers within 2x KLD Standard Error of the best performer, models are ranked by descending hidden-set accuracy (`hidden_mean`), descending CPU generation throughput, ascending file size, and ascending peak RSS.


| Candidate               | CPU Gen (t/s) | Peak RSS | Mean KLD | File Size |
| ----------------------- | ------------- | -------- | -------- | --------- |
| **ud_q4_k_xl (Winner)** | 16.81         | 3.12 GB  | 0.0289   | 2.73 GB   |
| q4_k_m_default          | 15.63         | 3.80 GB  | 0.0311   | 3.42 GB   |
| q4_k_m_ud_style         | 17.33         | 3.04 GB  | 0.0367   | 2.64 GB   |
| iq4_nl                  | 14.56         | 3.73 GB  | 0.0397   | 3.35 GB   |
| iq4_xs                  | 15.67         | 3.68 GB  | 0.0410   | 3.29 GB   |
| q4_0_imatrix_pure       | 16.79         | 3.03 GB  | 0.0499   | 2.64 GB   |
| q8_0_ceiling (Control)  | 11.45         | 5.29 GB  | 0.0013   | 4.95 GB   |


The winner, `ud_q4_k_xl`, provided an excellent balance of low memory footprint (~3.12 GB peak RSS), high CPU throughput (16.81 tokens/s), and acceptable KLD divergence (0.0289) relative to the 8-bit ceiling control (KLD 0.0013). This replaced an earlier Gate 1 candidate (`kuza-eng-sw-q4_k_m-imatrix-q6emb`) that used a different imatrix-heavy recipe.

`metadata.json` reports the quantization as `GGUF mixed Q4_K/Q5_K (imatrix)`, which matches the on-disk tensor mix rather than a single uniform Q4_K_M file.

---



## Model Provenance

This section matches the `provenance` object in `[metadata.json](metadata.json)` and the proof-of-training files in `[provenance/](provenance/)`.

- **Base model source:** `huggingface:unsloth/gemma-4-E2B-it-qat-q4_0-unquantized`
- **Base model commit SHA:** `13139d07f4cc4b6a2585ea6ab9fde6063281ae14` (upstream revision the training pipeline started from)
- **Download pin (submitted GGUF):** Hugging Face commit `ed650f8fc212de1bb4e4d2684b32809a7fb59b32` for `quants/ud_q4_k_xl/kuza-ud-q4_k_xl.gguf` in `[download_model.sh](download_model.sh)` — this is the immutable file evaluators fetch, not the base model revision
- **Fine-tuning method:** **QLoRA** — multi-dataset SFT, then DPO on preference pairs (Unsloth QAT int4 LoRA; see `[provenance/sft_manifest.json](provenance/sft_manifest.json)`)
- **Training datasets:** `[kuza_sft_english](https://huggingface.co/datasets/kuzaai/kuza_sft_english)`, `[kuza_sft_swahili](https://huggingface.co/datasets/kuzaai/kuza_sft_swahili)`, `[kuza_sft_multiturn](https://huggingface.co/datasets/kuzaai/kuza_sft_multiturn)`, `[kuza_sft_adversarial](https://huggingface.co/datasets/kuzaai/kuza_sft_adversarial)`, `[kuza_dpo_preference](https://huggingface.co/datasets/kuzaai/kuza_dpo_preference)` — revision SHAs and sizes in `[provenance/dataset_info.md](provenance/dataset_info.md)`
- **Automation:** [adtc-pipeline](https://github.com/sudouserx/adtc-pipeline/tree/main/gemma-4-e2b) `gemma-4-e2b` (scripts copied under `[provenance/scripts/](provenance/scripts/)`). The end-to-end workflow is orchestrated by `run.sh`, executing sequentially: dependency installation (`00_install_deps.py`), `llama.cpp` setup, SFT (`02_sft.py`), DPO (`02b_dpo.py`), reference generation, QAT-aware export (`03b_qat_export.py`), imatrix calculation (`04_imatrix.py`), quantization (`05_quants.py`), screening (`06_screen.py`), provenance packaging (`07_provenance.py`), and Hub upload (`08_upload.py`). Full artifacts on [kuzaai/kuza-gemma-4-e2b](https://huggingface.co/kuzaai/kuza-gemma-4-e2b)
- **Adapters committed:** `[provenance/adapter-sft/](provenance/adapter-sft/)` (post-SFT), `[provenance/adapter-dpo/](provenance/adapter-dpo/)` (post-DPO, checkpoint-109 before merge)
- **Checksums:** `[provenance/checksums.sha256](provenance/checksums.sha256)`

DPO training improved preference accuracy over steps (see reward/accuracy metrics in `[provenance/logs/dpo_trainer_state.json](provenance/logs/dpo_trainer_state.json)`).

### Before/after comparison

Base outputs are from **stock Gemma 4 E2B IT** (no Kuza adapter). Kuza outputs are from the submitted `kuza-ud-q4_k_xl.gguf` via `llama.cpp` with the Kuza agricultural system prompt. Prompts match `[metadata.json](metadata.json)` test prompts.

#### tp_001 — English (calf care)

**Prompt:** How do I take care of my calf for the first week to prevent mortality?


| Model                  | Output (excerpt)                                                                                                                                                                                             |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Base Gemma 4 E2B IT    | Treats “calf” as human leg injury: rest, elevation, ice packs, compression bandage, pain medication, “consult a doctor” — not livestock husbandry.                                                           |
| Kuza `kuza-ud-q4_k_xl` | Livestock guidance: dry/warm calf, **colostrum within the first hour**, clean bedding (~30 °C), twice-daily checks, vaccination/deworming per local vet schedule, pen hygiene, contact vet if illness signs. |




#### tp_002 — Swahili (banana spacing)

**Prompt:** Nipasheje umbali wa kupanda ndizi?


| Model                  | Output (excerpt)                                                                                                                                                                                      |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Base Gemma 4 E2B IT    | Long generic essay on varietals, climate, and trellis systems; suggests **~1.5–2.5 m** spacing without concrete row/plant layout for East African planting.                                           |
| Kuza `kuza-ud-q4_k_xl` | **3 m** between plants within a row and **3 m** between rows; thinning to one mat when ~1 m tall (~2 m between mats); drainage, organic matter, plant at nursery depth, plant at start of long rains. |


---



## Constraints



### Hardware and compute

The full data preparation and fine-tuning pipeline was developed on **free Kaggle T4 sessions with 16 GB VRAM and a 12-hour session limit**. These constraints affected model size, batch sizes, training duration, and the number of experiments that could be run.

To make the workflow robust to session termination, the pipeline was split into modular stages. Outputs from one stage could be reused by another notebook rather than requiring a single long-running job.

### Deployment

The target is CPU-only inference on a consumer laptop with approximately **8 GB RAM**. The submitted model therefore runs locally through `llama.cpp` with no external network calls, cloud APIs, retrieval services, or remote model dependencies.

The deployment constraint directly influenced the model choice, training-data compression strategy, and mixed-precision quantization approach.

---



## Benchmarks

The following are **participant-mode** measurements from `adtc-profiler` on the development laptop for `kuza-ud-q4_k_xl.gguf` (local `submission.json`, `measured_on: participant_laptop`, random seed 42, 4 threads). Official ADTC scores are measured separately on the standard evaluation environment.


| Metric               | Value                                  |
| -------------------- | -------------------------------------- |
| Machine              | AMD Ryzen 3 7320U with Radeon Graphics |
| GPU                  | None                                   |
| OS                   | Fedora Linux 43 (Workstation Edition)  |
| RAM                  | 7.0 GB                                 |
| Peak RSS             | 3.75 GB                                |
| Steady-state RSS     | 1.82 GB                                |
| Time to first token  | 11.30 s                                |
| Generation speed     | 16.79 tokens/s                         |
| Prompt length        | 512 tokens                             |
| Generated tokens     | 128                                    |
| Peak CPU temperature | 83.0 °C                                |
| Thermal throttling   | None observed                          |


Peak memory (~3.75 GB) and generation throughput (16.79 tokens/s) indicate the submitted quant fits the 8 GB laptop profile with higher throughput than the earlier Gate 1 artifact, at the cost of higher first-token latency on this CPU.

The profiler run did not include a quantitative accuracy benchmark (`accuracy` empty). Expert evaluation of factuality and safety, particularly for agricultural treatment and dosage recommendations, remains an important next step.

---



## Links and Resources

All relevant repositories, datasets, and artifacts referenced in this technical report are listed below.

### Repositories & Automation

- **ADTC Pipeline (Main):** [https://github.com/sudouserx/adtc-pipeline](https://github.com/sudouserx/adtc-pipeline)
- **ADTC Pipeline (Gemma 4 E2B Implementation):** [https://github.com/sudouserx/adtc-pipeline/tree/main/gemma-4-e2b](https://github.com/sudouserx/adtc-pipeline/tree/main/gemma-4-e2b)



### Datasets

**Source Corpora:**

- **Digital Green FarmerChat (Large):** [https://huggingface.co/datasets/DigiGreen/farmerchat-queries-large](https://huggingface.co/datasets/DigiGreen/farmerchat-queries-large)
- **Priority Queries Sample:** [https://huggingface.co/datasets/kuzaai/priority_queries](https://huggingface.co/datasets/kuzaai/priority_queries)
- **Long-tail Queries Sample:** [https://huggingface.co/datasets/kuzaai/longtail_queries](https://huggingface.co/datasets/kuzaai/longtail_queries)

**Kuza Fine-Tuning Datasets:**

- **Kuza SFT English:** [https://huggingface.co/datasets/kuzaai/kuza_sft_english](https://huggingface.co/datasets/kuzaai/kuza_sft_english)
- **Kuza SFT Swahili:** [https://huggingface.co/datasets/kuzaai/kuza_sft_swahili](https://huggingface.co/datasets/kuzaai/kuza_sft_swahili)
- **Kuza SFT Adversarial:** [https://huggingface.co/datasets/kuzaai/kuza_sft_adversarial](https://huggingface.co/datasets/kuzaai/kuza_sft_adversarial)
- **Kuza SFT Multi-turn:** [https://huggingface.co/datasets/kuzaai/kuza_sft_multiturn](https://huggingface.co/datasets/kuzaai/kuza_sft_multiturn)
- **Kuza DPO Preference:** [https://huggingface.co/datasets/kuzaai/kuza_dpo_preference](https://huggingface.co/datasets/kuzaai/kuza_dpo_preference)



### Model Artifacts & Results

- **Full Pipeline Artifacts (Hugging Face):** [https://huggingface.co/kuzaai/kuza-gemma-4-e2b](https://huggingface.co/kuzaai/kuza-gemma-4-e2b)
- **Selected Model Download (GGUF):** [https://huggingface.co/kuzaai/kuza-gemma-4-e2b/resolve/main/quants/ud_q4_k_xl/kuza-ud-q4_k_xl.gguf](https://huggingface.co/kuzaai/kuza-gemma-4-e2b/resolve/main/quants/ud_q4_k_xl/kuza-ud-q4_k_xl.gguf)
- **Full Pipeline Code:** [https://github.com/sudouserx/adtc-pipeline/tree/main/gemma-4-e2b](https://github.com/sudouserx/adtc-pipeline/tree/main/gemma-4-e2b)
- **Screening Results (**`results.json`**):** [https://huggingface.co/kuzaai/kuza-gemma-4-e2b/blob/main/screen/results.json](https://huggingface.co/kuzaai/kuza-gemma-4-e2b/blob/main/screen/results.json)

