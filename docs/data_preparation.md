# Data preparation

Use the scripts below to download and prepare each dataset. Follow the corresponding licenses and terms. All default output paths are ignored by git.

## FineWeb-Edu 100B

The FineWeb recipe uses the `sample-100BT` subset of `HuggingFaceFW/fineweb-edu`, GPT-2 tokenization, a leading `<|endoftext|>` token for every document, and a continuous uint16 token stream.

```bash
bash scripts/download_fineweb.sh
```

The download wrapper calls the preparation script with these defaults:

```bash
python scripts/prepare_fineweb.py \
  --subset sample-100BT \
  --output-dir data/fineweb-edu100B \
  --shard-size 100000000
```

### Binary layout

Each shard contains:

1. a 256-element int32 header (1,024 bytes);
2. `header[0:3] = [20240520, 1, num_tokens]`;
3. `num_tokens` uint16 GPT-2 token IDs.

The first 100M-token shard is `fineweb_val_000000.bin`; subsequent shards are training files. `FineWebShards` skips 512 uint16 values when reading, exactly matching the 256-int32 header. The vocabulary is padded to 50,304 in the model, while token IDs remain those of GPT-2.

Preparation is streamed and parallelized, but the complete tokenized corpus is large. Plan disk space for roughly 100 GB of raw uint16 tokens per 50B tokens, plus validation and working storage; the full recipe uses approximately 49.15B training tokens and also writes a validation shard. Actual source and working-space requirements can be larger.

For a small tokenizer/shard test, use `--max-documents`. To test both train and validation loaders with a small corpus, also choose a small `--shard-size` large enough to hold a useful number of 1,025-token windows. These smoke-test shards are not suitable for reproducing training results.

## C4 RealNewsLike for GPT-2

Download C4 RealNewsLike from Hugging Face and save the raw dataset locally:

```bash
python scripts/download_c4.py \
  --output-path data/raw/c4-realnewslike
```

The script records the source, split, row count, and columns in `download_meta.json`. Then build the fixed-length cache once and share it across the QK parameter-sharing comparison methods:

```bash
python scripts/prepare_c4.py \
  --input-path data/raw/c4-realnewslike \
  --output-path data/cache/c4-realnewslike-gpt2-1024 \
  --tokenizer gpt2 \
  --seq-length 1024 \
  --val-fraction 0.005 \
  --num-proc 32 \
  --seed 42
```

The script:

1. makes a deterministic train/validation split;
2. tokenizes without adding special tokens;
3. concatenates token streams within each processing batch;
4. emits fixed 1,024-token chunks;
5. saves a `DatasetDict` with `train` and `val` plus `meta.json`.

The paper cache contains approximately 7.73M training chunks and 38.9K validation chunks. Counts can vary if the raw corpus snapshot or preprocessing differs, so record `meta.json` with each reproduction.

`--overwrite` deletes an existing output cache. Use it only when intentionally rebuilding the cache.

## ImageNet-1K

Download the official ILSVRC2012 training/validation archives and devkit through the [ImageNet download page](https://image-net.org/download.php). The preparation script uses torchvision's official devkit and validation parsing logic:

```bash
python scripts/prepare_imagenet.py \
  --archive-root /path/to/archives \
  --output-root data/imagenet1k \
  --train-dir /path/to/extracted/train
```

Expected archive names are:

```text
ILSVRC2012_img_train.tar
ILSVRC2012_img_val.tar
ILSVRC2012_devkit_t12.tar.gz
```

The resulting layout is an ImageFolder tree:

```text
data/imagenet1k/
  train/<wnid>/*.JPEG
  val/<wnid>/*.JPEG
  meta.bin
```

If `train/` already exists under `--output-root`, `--train-dir` is unnecessary. `--skip-val-prepare` only links available inputs and skips validation extraction. The script refuses to replace unrelated existing links or a partially populated validation directory.

Training uses 224×224 images, standard ImageNet normalization, and resize to 256 for evaluation. Mixup, CutMix, label smoothing, and optimization settings are recorded in `configs/imagenet_main.yaml` and `configs/experiments/*.json`.
