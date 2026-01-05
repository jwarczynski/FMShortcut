# One-step Nonautoregressive Natural Language Generation with Shortcut Flow Matching Models

> Official Codebase for [*One-step Nonautoregressive Natural Language Generation with Shortcut Flow Matching Models*].

## Datasets

Prepare datasets and put them under the `datasets` folder. We use three datasets in our paper.

| Dataset | Task | Source | HuggingFace ID |
|---------|------|--------|----------------|
| QQP-Official | Paraphrase Generation | [download](https://drive.google.com/drive/folders/1D6PxrfB1410XFJVGnbXR5bGhb-ulIX_l?usp=sharing) | — |
| ParaSCI | Scientific Paraphrase | [paper](https://aclanthology.org/2021.acl-long.382/) | [`HHousen/ParaSCI`](https://huggingface.co/datasets/HHousen/ParaSCI) |
| PAWS-Wiki | Paraphrase Identification | [GitHub](https://github.com/google-research-datasets/paws) | [`google-research-datasets/paws`](https://huggingface.co/datasets/google-research-datasets/paws) (config: `labeled_final`) |

### Tokenization

Tokenize datasets before training using `scripts/run_tokenize.py`:

```bash
# QQP (from local jsonl files):
uv run python scripts/run_tokenize.py --dataset QQP --data_dir datasets/QQP-Official --tokenizer bert-base-uncased --max_seq_length 128

# ParaSCI (downloaded automatically from HuggingFace):
uv run python scripts/run_tokenize.py --dataset parasci --tokenizer bert-base-uncased --max_seq_length 256

# PAWS-Wiki (from local TSV files, filtered for paraphrases with label=1):
uv run python scripts/run_tokenize.py --dataset paws_wiki --data_dir datasets/paws_wiki --tokenizer bert-base-uncased --max_seq_length 128
```

Tokenized datasets are saved under `datasets/tokenized/{tokenizer_name}/{dataset_name}/`.

## Training

For training, run:

```bash
# QQP shortcut model:
uv run python -m shortcutfm configs/training/scut.yaml

# QQP baseline (flow-matching only):
uv run python -m shortcutfm configs/training/baseline.yaml
```

Training configs for other datasets can be found in `configs/training/`.

## Generation & Evaluation

Generate texts and evaluate with multiple NFE values:

```bash
uv run python -m shortcutfm.decoding.generate configs/generation/qqp.yaml
```

## Citation

Please add the citation if our paper or code helps you.

```tex

```
