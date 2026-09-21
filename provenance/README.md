# Proof-of-training — Kuza AI (Gate 2)

This folder documents how **`kuza-ud-q4_k_xl.gguf`** was produced from `unsloth/gemma-4-E2B-it-qat-q4_0-unquantized`.

| Path | Contents |
|------|----------|
| [`ADAPTER_WEIGHTS_AND_LFS.md`](ADAPTER_WEIGHTS_AND_LFS.md) | Why adapter `.safetensors` may be off GitHub, HF mirror URLs, LFS upload steps |
| [`adapter-sft/`](adapter-sft/) | QLoRA weights after multi-dataset SFT |
| [`adapter-dpo/`](adapter-dpo/) | QLoRA weights after DPO (merged into reference before quant) |
| [`logs/`](logs/) | SFT metrics + DPO `trainer_state.json` loss/reward curves |
| [`scripts/`](scripts/) | Training/merge/quant scripts copied from [adtc-pipeline](https://github.com/sudouserx/adtc-pipeline) @ `7463737c22551bd99e2314771ea7958bb29cc974` |
| [`dataset_info.md`](dataset_info.md) | Dataset URLs, sizes, revision SHAs, licenses |
| [`checksums.sha256`](checksums.sha256) | SHA256 for base revision, merged ref, GGUF, adapters |
| [`merge_quant_notes.md`](merge_quant_notes.md) | Stage-by-stage merge and quantization narrative |
| [`sft_manifest.json`](sft_manifest.json) | Full SFT/DPO hyperparameters and dataset revisions |
| [`screen_results.json`](screen_results.json) | Quant variant screening (winner: `ud_q4_k_xl`) |
| [`ud_q4_k_xl_recipe.json`](ud_q4_k_xl_recipe.json) | Exact llama.cpp quant recipe for submitted GGUF |

Large binary artifacts also live on Hugging Face: https://huggingface.co/kuzaai/kuza-gemma-4-e2b

**Git LFS:** `adapter_model.safetensors` files (~184 MB each) are listed in [`.gitattributes`](../.gitattributes) for LFS. If they are not yet on GitHub, see [`ADAPTER_WEIGHTS_AND_LFS.md`](ADAPTER_WEIGHTS_AND_LFS.md) — identical files are on [kuza-gemma-4-e2b](https://huggingface.co/kuzaai/kuza-gemma-4-e2b) at the same revision as `download_model.sh`.
