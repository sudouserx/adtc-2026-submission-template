# Training datasets — Kuza Gemma 4 E2B

All datasets are published on Hugging Face under `kuzaai/`. Revision SHAs below match [`sft_manifest.json`](sft_manifest.json) (`dataset_revisions`).

| Dataset | HF URL | Role | Approx. size | Revision SHA | License |
|---------|--------|------|--------------|--------------|---------|
| `kuza_sft_english` | https://huggingface.co/datasets/kuzaai/kuza_sft_english | Primary English agricultural SFT (~25.6K curated examples after filtering) | ~25.6K rows | `64fdd984a2c897a74f12d3245d9905e2bbfb4e98` | MIT (see dataset card) |
| `kuza_sft_swahili` | https://huggingface.co/datasets/kuzaai/kuza_sft_swahili | Swahili agricultural SFT (translated from English corpus) | paired with EN mix | `893ad3614ef922c3fa6948f41279340f6cb8da4a` | MIT (see dataset card) |
| `kuza_sft_multiturn` | https://huggingface.co/datasets/kuzaai/kuza_sft_multiturn | Multi-turn farm advisory dialogues | subset in SFT mix | `fd1dafa120271463eb5997caf0ae237b3ce4feb2` | MIT (see dataset card) |
| `kuza_sft_adversarial` | https://huggingface.co/datasets/kuzaai/kuza_sft_adversarial | Safety / refusal-oriented examples | ~117–1.2K in mix | `a37377a7239ecf6ca2753ba5a0d10d01db6d3cac` | MIT (see dataset card) |
| `kuza_dpo_preference` | https://huggingface.co/datasets/kuzaai/kuza_dpo_preference | Preference pairs for DPO (1100 train / 122 eval pairs) | 1,224 rows total | see DPO dataset card | MIT |

Additional SFT mixing used a small **general** split (revision `e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b`) documented in `sft_manifest.json`.

## Provenance of raw agricultural text

The English SFT corpus was derived from the Digital Green FarmerChat large dataset (~1.45M farmer Q&A pairs), filtered and densified via an offline preparation pipeline (see [`REPORT.md`](../REPORT.md)). Teacher models were used **only during dataset construction**, not at inference.

## Representative samples

- [`data/sample_heldout.jsonl`](data/sample_heldout.jsonl) — three held-out English farm Q&A rows (from HF `adapter-dpo/heldout.jsonl`).
- Full datasets remain on Hugging Face; checksums of dataset revisions are pinned in `sft_manifest.json`.
