# Adapter weights, Git LFS, and Hugging Face mirror

Gate 2 requires proof-of-training, including QLoRA adapter weights (`adapter_model.safetensors`). Each file is about **184 MB** (193,326,648 bytes), so GitHub requires **Git LFS** for hosting in this repository.

## Why adapter files may be missing on GitHub

Early commits listed `provenance/adapter-dpo/adapter_model.safetensors` and `provenance/adapter-sft/adapter_model.safetensors` in [`.gitignore`](../.gitignore), so Git never staged them even though [`.gitattributes`](../.gitattributes) defines LFS tracking. As a result, **configs, logs, and scripts** under `provenance/adapter-*` can be on GitHub while **weight files are not**.

After fixing `.gitignore` and running the procedure below, GitHub should serve LFS pointer files; the binary payload lives on GitHub’s LFS storage.

## Canonical copies on Hugging Face

Same revision as the submitted GGUF pin in [`download_model.sh`](../download_model.sh) (`ed650f8fc212de1bb4e4d2684b32809a7fb59b32`):

| Artifact | Download URL |
|----------|--------------|
| DPO adapter (post–checkpoint-109) | https://huggingface.co/kuzaai/kuza-gemma-4-e2b/resolve/ed650f8fc212de1bb4e4d2684b32809a7fb59b32/adapter-dpo/adapter_model.safetensors |
| SFT adapter | https://huggingface.co/kuzaai/kuza-gemma-4-e2b/resolve/ed650f8fc212de1bb4e4d2684b32809a7fb59b32/adapter-sft/adapter_model.safetensors |

Full artifact tree: https://huggingface.co/kuzaai/kuza-gemma-4-e2b

### SHA256 verification

Compare downloads to [`checksums.sha256`](checksums.sha256):

```text
b7f47b3ea368e7bd0ff6cad3f31e7147b144042c028f8e57b1cf6a330ff00747  provenance/adapter-dpo/adapter_model.safetensors
feafdea580dde231950fc625a8ebe5cd3010c3577e4737675c26ca2e1afe6f4d  provenance/adapter-sft/adapter_model.safetensors
```

```bash
sha256sum provenance/adapter-dpo/adapter_model.safetensors
sha256sum provenance/adapter-sft/adapter_model.safetensors
```

## Maintainer: add adapters to GitHub via LFS

Run from the repository root:

```bash
git lfs install
git lfs track "provenance/**/adapter_model.safetensors"
git add -f provenance/adapter-dpo/adapter_model.safetensors
git add -f provenance/adapter-sft/adapter_model.safetensors
git add .gitattributes .gitignore provenance/ADAPTER_WEIGHTS_AND_LFS.md provenance/README.md
git commit -m "Add QLoRA adapter weights via Git LFS and document HF mirror."
git push origin main
git lfs push origin main --all
```

Use `git add -f` only if a local ignore rule still matches during transition.

## Verification

1. **Local LFS tracking**

   ```bash
   git lfs ls-files
   ```

   Expect two paths ending in `adapter_model.safetensors`.

2. **Pointer on GitHub (raw)**

   ```bash
   curl -sL "https://raw.githubusercontent.com/sudouserx/adtc-2026-submission-template/main/provenance/adapter-dpo/adapter_model.safetensors" | head -3
   ```

   Expect:

   ```text
   version https://git-lfs.github.com/spec/v1
   oid sha256:b7f47b3ea368e7bd0ff6cad3f31e7147b144042c028f8e57b1cf6a330ff00747
   ```

3. **LFS media URL** (after `git lfs push` succeeds)

   ```bash
   curl -sI "https://media.githubusercontent.com/media/sudouserx/adtc-2026-submission-template/main/provenance/adapter-dpo/adapter_model.safetensors" | head -5
   ```

   Expect HTTP **200** (not 404).

If GitHub LFS is unavailable, reviewers can use the Hugging Face URLs above and verify checksums.
