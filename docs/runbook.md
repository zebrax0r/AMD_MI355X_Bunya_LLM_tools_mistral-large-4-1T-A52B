# Operating runbook

[Documentation home](../README.md) | [Configuration](configuration.md) | [Troubleshooting](troubleshooting.md)

## 1. Checkout and configuration (login node)

```bash
cd "/scratch/user/$USER/builds"
git clone <this repo> AMD_MI355X_Bunya_LLM_tools_mistral-large-4-1T-A52B
cd AMD_MI355X_Bunya_LLM_tools_mistral-large-4-1T-A52B
cp -n ml4-env.example ml4.env && chmod 600 ml4.env     # defaults work; STATE_DIR is per-user scratch
python3 -m unittest discover -s tests                   # CPU-only sanity
```

The weights are already in the shared store ([weights](weights.md)). You do
not download anything unless a checkpoint is missing.

## 2. Batch (recommended)

```bash
./submit-ml4.sh mi355x                   # 8 x MI355X: MXFP4, TP4 + monokernel (AGENTS<=4) or TP8
./submit-ml4.sh mi355x --gpus 4          # 4 x MI355X: the same TP4 layout, half the node
./submit-ml4.sh mi355x --gateway litellm # 8 x MI355X: 2 x TP4 replicas behind LiteLLM
./submit-ml4.sh mi355x --agents 16       # 8 x MI355X: TP8 + DSpark, 64 sequences
./submit-ml4.sh mi300x                   # 8 x MI300X: FP8 TP8 (branch engine)
grep -A 25 'is up on' logs/ml4-<jobid>.out
```

Account and QoS default to the K3 repos' values (`a_rcc` / `sdf`). Pass others
after `--`: `./submit-ml4.sh mi300x -- --account=a_xyz --qos=normal`.

## 3. Interactive

```bash
salloc --partition=gpu_rocm --account=a_rcc --qos=sdf --nodes=1 \
       --gres=gpu:mi355x:8 --cpus-per-task=96 --mem=1500G --time=24:00:00
srun --jobid "$SLURM_JOB_ID" --pty /bin/bash -l      # if salloc left you on the login node

./serve-ml4.sh plan            # resolved recipe + argv; no GPU touched
./serve-ml4.sh pull            # once per image: ~9.5 GB (+ LiteLLM ~1 GB with 'pull all')
./serve-ml4.sh gpucheck        # expect: rocm OK, 8 x gfx950 (or gfx942), 256 CUs on MI355X
./serve-ml4.sh check           # vllm imports, registry knows MistralLarge4, levers present
./serve-ml4.sh flagcheck       # the image accepts every flag of this recipe
./serve-ml4.sh serve --detach  # or: GATEWAY=litellm ./serve-ml4.sh serve --detach
```

On MI300X the first `serve` (or `./serve-ml4.sh overlay`) also prepares the
branch overlay. This is a one-time copy into `STATE_DIR`, about 1 minute.

## 4. Verify

```bash
./serve-ml4.sh status          # processes, endpoint, DSpark acceptance, throughput
./serve-ml4.sh toolcheck       # 2-turn tool call with reasoning replay
./serve-ml4.sh vibecheck       # the same in Mistral Vibe's wire format
./agentstress.py http://127.0.0.1:8000 "$(cat $STATE_DIR/ml4-api-key)" mistral-large-4 150000 20000
./bench-ml4.sh                 # random-prompt sweep (concurrency 1 2 4 8)
```

`/health` answering does not mean the output is coherent. Read an actual reply.

## 5. Connect clients, stop

See [clients](clients.md) (tunnel, Mistral Vibe, LiteLLM). Stop with
`./serve-ml4.sh stop` on the node, or `scancel <jobid>`. Logs stay in
`$STATE_DIR/logs/`, and time-to-ready history in `$STATE_DIR/ml4-loadtimes.log`.
