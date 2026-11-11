# Shared weights store

[Documentation home](../README.md) | [Runbook](runbook.md)

The weights live in `/scratch/opendata/models`. Any Bunya user can read them,
so nobody needs a personal 1-1.6 TB copy. Each user's own state (images,
caches, keys, logs) lives in `STATE_DIR` (default
`/scratch/user/$USER/models/mistral-large-4`).

| Checkpoint | Path | Pin | Size | Serves on |
|---|---|---|---|---|
| `mistralai/Mistral-Large-4-1T-A52B` (official FP8, block 128x128) | `/scratch/opendata/models/gguf/mistral-large-4` | `af26e786` (2 Nov 2026) | 1,061 GB, 54 shards | MI300X, MI355X |
| `amd/Mistral-Large-4-1T-A52B-Quark-MXFP4` (AMD Quark, OCP MX FP4 experts) | `/scratch/opendata/models/huggingface/amd-Mistral-Large-4-1T-A52B-Quark-MXFP4` | `d084803a` (29 Oct 2026) | 579 GB (539 GiB), 54 shards | MI355X only |

Both use the Mistral-native layout: `params.json`, `tekken.json`,
`consolidated-*.safetensors` and `consolidated.safetensors.index.json`. Both
also carry the HF-format `config.json`. The launcher serves them with
`--tokenizer-mode/--config-format/--load-format mistral`. Each embeds the
DSpark draft head (`mtp_dspark.*`), which stays in BF16 in both.

Notes:

- The FP8 directory is under `gguf/` because that is where it was asked to go.
  It holds safetensors, not GGUF.
- The MXFP4 copy was already in the store (owner `uqsarahw`). It matches the
  Hub byte for byte at `d084803a`. AMD's later commits changed only the
  README.
- `BUNYA-MANIFEST.json` in each directory records what was verified and when.

## Check or fetch

```bash
./serve-ml4.sh verify-weights all     # index vs files on disk, and sizes vs the Hub at the pinned commit
./serve-ml4.sh download fp8           # resumable; no-op when already complete
./serve-ml4.sh download mxfp4
```

`download` uses `umask 022`, then runs `chmod -R go+rX`, so other users can
read the files. It needs write access to the target directory. A complete
directory is never re-downloaded. `serve` refuses to start on an incomplete
directory, which avoids wasting a half-hour load.

## Using another location

Set `FP8_PATH` / `MXFP4_PATH` in `ml4.env`. Any directory with the files above
works, including a personal copy.
