# Merge and quantization path

Submitted artifact: **`kuza-ud-q4_k_xl.gguf`** (pipeline recipe name **`ud_q4_k_xl`**: imatrix-guided **mixed Q4_K / Q5_K** GGUF; reported in metadata as `GGUF mixed Q4_K/Q5_K (imatrix)`).

## Pipeline stages

1. **Base model** — `unsloth/gemma-4-E2B-it-qat-q4_0-unquantized` @ revision `13139d07f4cc4b6a2585ea6ab9fde6063281ae14`.
2. **QLoRA SFT** — [`scripts/02_sft.py`](scripts/02_sft.py); adapter in [`adapter-sft/`](adapter-sft/); config in [`sft_manifest.json`](sft_manifest.json).
3. **QLoRA DPO** — [`scripts/02b_dpo.py`](scripts/02b_dpo.py) on `kuzaai/kuza_dpo_preference`; final adapter [`adapter-dpo/`](adapter-dpo/) checkpoint-109 (see `trainer_state.json`).
4. **Merge + reference export** — [`scripts/03_reference.py`](scripts/03_reference.py) produces merged BF16 reference (SHA256 `55cb84f3…` in [`ud_q4_k_xl_recipe.json`](ud_q4_k_xl_recipe.json)).
5. **Imatrix + quant candidates** — imatrix collection and [`scripts/05_quants.py`](scripts/05_quants.py) build multiple GGUF variants.
6. **Screening** — [`screen_results.json`](screen_results.json) selects **`ud_q4_k_xl`** (winner: best gate-passing balance of hidden-prompt behavior, CPU throughput, size, and KL vs Q8 ceiling).

## UD-Q4_K_XL recipe (summary)

See [`scripts/ud_q4_k_xl_recipe.json`](scripts/ud_q4_k_xl_recipe.json):

- Bulk FFN tensors: Q4_K
- Attention Q/K/V/O and selected projections: Q5_K
- Token embeddings / output tensor: Q5_K overrides
- Uses imatrix SHA256 `118e886c…`
- Output file SHA256 `79e604f9…`, size 2,725,270,560 bytes

## Automation source repo

Training/quant automation: https://github.com/sudouserx/adtc-pipeline/tree/main/gemma-4-e2b  
Pinned commit for copied scripts: **`7463737c22551bd99e2314771ea7958bb29cc974`**

Full artifact bundle (adapters, merged weights, all quants): https://huggingface.co/kuzaai/kuza-gemma-4-e2b
