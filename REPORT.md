# Technical Report — Kuza AI: Offline Agricultural AI for East African Farmers

**Team ID:** kuza-ai
**Domain:** agriculture
**Model:** kuza-eng-sw-q4_k_m-imatrix-q6emb

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
| --------------------------- | ---------------: |
| Gemma 4 E4B IT              |    2.24 tokens/s |
| DeepSeek-R1-Distill-Qwen-7B |    1.46 tokens/s |
| Gemma 4 E2B IT              |    4.70 tokens/s |
| Phi-4                       |    3.60 tokens/s |

Gemma 4 E2B provided the best throughput among the evaluated candidates while maintaining a relatively small model footprint. I therefore started from the Unsloth `gemma-4-E2B-it-qat-q4_0-unquantized` checkpoint. The checkpoint is QAT-oriented and provides a multimodal foundation for possible future image and audio agricultural use cases.

### Fine-tuning data

The primary objective of the data pipeline was to maximize **useful agricultural information per training example**, rather than simply maximizing dataset size. This was important because the final model needed to remain small enough for CPU-only, offline laptop inference.

The pipeline started from the Digital Green FarmerChat large dataset containing approximately **1.45 million farmer-initiated Q&A pairs**, including substantial African data such as Kenya and Ethiopia.

The preparation pipeline was:

1. Analyze the FarmerChat corpus and distinguish high-priority/frequent questions from long-tail questions, while using semantic representations to group related queries and improve sampling diversity.
2. Sample **50K priority examples and 20K long-tail examples** for further processing.
3. Use **Llama-3.3-70B-Versatile** to clean the questions and answers, remove filler, and increase information density.
4. Produce an approximately **56K-example intermediate corpus**.
5. From this corpus, remove malformed examples, duplicates, and examples whose answers were too semantically similar to other answers. The resulting final English training set contains approximately **25.6K examples**.
6. Generate a corresponding Swahili SFT dataset from the English corpus using **GPT-OSS-120B**.
7. Add a small adversarial dataset for safety-oriented training examples.

The removed examples remain available in the intermediate `agri_sft_prod_56k` dataset but are not used for training the submitted model.

Large teacher models were used only during offline dataset preparation. The final submitted GGUF does not require these models or any external API during inference.

### Alternatives considered

I first explored constructing a dataset directly from approximately 200 agricultural extension guidelines, handbooks, and other public sources by extracting facts and reverse-generating conversations. This produced less useful data because the available material was difficult to gather systematically, was not sufficiently dense at the required scale, and generated questions were less representative of real farmer interactions than farmer-initiated queries.

I also investigated the two-stage agricultural advisory approach described in *Fine-Tuning and Evaluating Conversational AI for Agricultural Advisory*, where factual knowledge handling and conversational response generation are separated. This is a promising direction for a production system, but the ADTC submission requires a single GGUF model running through `llama.cpp`, so that architecture was not used in the submitted system.

### Quantization

The final model uses a mixed-precision GGUF configuration:

* **Q4_K_M** for the main model weights
* **Q6_K** for selected attention projection and embedding-related tensors
* **Q8_0** for selected embedding/language-model-head tensors

I evaluated pure Q4_0 and Q5_0 variants as additional alternatives. Quantization variants were compared using **KL divergence and generation throughput**.

Although the starting checkpoint was QAT-oriented, my experiments showed that using an **importance matrix (`imatrix`)** produced a better quality/efficiency trade-off than naïve quantization. The final submission therefore uses an `imatrix`-guided mixed-precision configuration.

---

## Model Provenance

<!-- Where did your model actually come from? This must match the `provenance` object in metadata.json. -->

- **Base model source:** e.g. `huggingface:microsoft/Phi-3-mini-4k-instruct-gguf`
- **Base model commit SHA:** the exact commit you started from (see README's Model Provenance section for how this differs from the download-pin commit in `download_model.sh`)
- **Fine-tuning method:** `none` / `prompt_engineering` / `lora` / `qlora` / `full_fine_tune`
- **Training datasets:** name(s) and source(s), or "N/A — used stock model as-is"

If you fine-tuned (method is not `none`), include a before/after comparison showing what changed:

<!-- e.g. a short table or 2-3 example prompts with the base model's output vs. your fine-tuned model's output, or a metric that moved (accuracy on a held-out set, etc.) -->

If you used a stock/off-the-shelf model without modification, say so plainly here — that's a legitimate and expected submission path; this section only needs the base model source and commit.

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

The following are self-reported participant-mode measurements from the development laptop. Official ADTC scores are measured separately by the ADTC profiler on its standard evaluation environment.

| Metric               | Value                                  |
| -------------------- | -------------------------------------- |
| Machine              | AMD Ryzen 3 7320U with Radeon Graphics |
| GPU                  | None                                   |
| OS                   | Fedora Linux 43 Workstation Edition    |
| RAM                  | 7.0 GB                                 |
| Peak RSS             | 4.94 GB                                |
| Steady-state RSS     | 0.70 GB                                |
| Time to first token  | 5.81 s                                 |
| Generation speed     | 5.47 tokens/s                          |
| Prompt length        | 512 tokens                             |
| Generated tokens     | 128                                    |
| Peak CPU temperature | 82.0 °C                                |
| Thermal throttling   | None observed                          |

The main measured limitation is **latency rather than memory**. Generation throughput was 5.47 tokens/s, while peak resident memory remained approximately 4.94 GB on a 7 GB RAM participant laptop. The benchmark therefore demonstrates that the model can run within the intended low-memory CPU environment, although further optimization is needed to reduce first-token latency.

The profiler submission currently contains no quantitative accuracy result, so no accuracy score is claimed in this report. Expert evaluation of factuality and safety, particularly for agricultural treatment and dosage recommendations, remains an important next step.
