#!/usr/bin/env bash
#
# vibe-setup.sh — point Mistral Vibe (github.com/mistralai/mistral-vibe) at
# Mistral Large 4 served from Bunya, directly (vLLM) or through LiteLLM.
# Run it on the machine where you use Vibe (your laptop), not on Bunya.
#
#   ./vibe-setup.sh --via direct  --host localhost --port 8000 --api-key "$KEY"   # ssh tunnel to vLLM
#   ./vibe-setup.sh --via litellm --host localhost --port 8000 --api-key "$KEY"   # ssh tunnel to the node's LiteLLM
#   ./vibe-setup.sh --via litellm --base https://llm.example.org/v1 --model mistral-large-4 --api-key "$KEY"
#   ./vibe-setup.sh --remove
#
# What it writes (VIBE_HOME, default ~/.vibe):
#   config.toml  a managed [[providers]] + [[models]] block (re-running replaces it)
#                and, unless --no-activate, the root key active_model
#   .env         BUNYA_ML4_API_KEY=<key>   (mode 600; Vibe loads it at start)
#
# Why api_style "openai" and not the model card's "reasoning": Vibe's
# "reasoning" style reads thinking only from Mistral-API content blocks
# ({"type": "thinking"}), while vLLM returns it in a separate field. With
# "openai" + reasoning_field_name Vibe shows the thinking and replays it on the
# next turn, which the model needs during tool use. vLLM names the field
# "reasoning"; LiteLLM normalises it to "reasoning_content".

set -euo pipefail

VIA="direct"; HOST="localhost"; PORT="8000"; BASE=""; KEY="${ML4_API_KEY:-}"; KEY_FILE=""
MODEL="mistral-large-4"; ALIAS="mistral-large-4-bunya"; PROVIDER="bunya-ml4"
CONTEXT="262144"; ACTIVATE=1; REMOVE=0; FIELD=""
VIBE_HOME_DIR="${VIBE_HOME:-$HOME/.vibe}"

usage() { sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
while (( $# )); do
    case "$1" in
        --via)        VIA="$2"; shift ;;
        --host)       HOST="$2"; shift ;;
        --port)       PORT="$2"; shift ;;
        --base)       BASE="$2"; shift ;;
        --api-key)    KEY="$2"; shift ;;
        --key-file)   KEY_FILE="$2"; shift ;;
        --model)      MODEL="$2"; shift ;;
        --alias)      ALIAS="$2"; shift ;;
        --provider)   PROVIDER="$2"; shift ;;
        --context)    CONTEXT="$2"; shift ;;
        --reasoning-field) FIELD="$2"; shift ;;
        --vibe-home)  VIBE_HOME_DIR="$2"; shift ;;
        --no-activate) ACTIVATE=0 ;;
        --remove)     REMOVE=1 ;;
        -h|--help)    usage 0 ;;
        *) echo "Unknown option: $1" >&2; usage 1 ;;
    esac
    shift
done
case "$VIA" in direct|litellm) ;; *) echo "--via must be direct or litellm" >&2; exit 1 ;; esac
[[ "$CONTEXT" =~ ^[0-9]+$ ]] || { echo "--context must be an integer" >&2; exit 1; }
[[ -n "$KEY_FILE" ]] && KEY="$(<"$KEY_FILE")"
[[ -n "$BASE" ]] || BASE="http://$HOST:$PORT/v1"
[[ -n "$FIELD" ]] || { [[ "$VIA" == litellm ]] && FIELD="reasoning_content" || FIELD="reasoning"; }
command -v python3 >/dev/null 2>&1 || { echo "python3 is required." >&2; exit 1; }
mkdir -p "$VIBE_HOME_DIR"

VIBE_HOME_DIR="$VIBE_HOME_DIR" VIA="$VIA" BASE="$BASE" KEY="$KEY" MODEL="$MODEL" ALIAS="$ALIAS" \
PROVIDER="$PROVIDER" CONTEXT="$CONTEXT" ACTIVATE="$ACTIVATE" REMOVE="$REMOVE" FIELD="$FIELD" \
python3 - <<'PY'
import os, re, stat
E = os.environ
home = E["VIBE_HOME_DIR"]
cfg_path, env_path = os.path.join(home, "config.toml"), os.path.join(home, ".env")
BEGIN, END = "# >>> bunya-ml4 (managed by vibe-setup.sh) >>>", "# <<< bunya-ml4 <<<"
ACT = "# bunya-ml4: active_model set by vibe-setup.sh"
KEYVAR = "BUNYA_ML4_API_KEY"

text = open(cfg_path).read() if os.path.exists(cfg_path) else ""
# drop our previous block and our active_model line
text = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n?", "", text, flags=re.S)
text = re.sub(r"(?m)^" + re.escape(ACT) + r"\n^active_model\s*=.*\n?", "", text)

if E["REMOVE"] != "1":
    ctx = int(E["CONTEXT"])
    block = f'''{BEGIN}
# Mistral Large 4 on Bunya ({E["VIA"]}). Re-run vibe-setup.sh to change; edits inside are overwritten.
[[providers]]
name = "{E["PROVIDER"]}"
api_base = "{E["BASE"]}"
api_key_env_var = "{KEYVAR}"
api_style = "openai"
backend = "generic"
reasoning_field_name = "{E["FIELD"]}"

[[models]]
name = "{E["MODEL"]}"
provider = "{E["PROVIDER"]}"
alias = "{E["ALIAS"]}"
temperature = 1.0
thinking = "high"
thinking_levels = ["low", "medium", "high"]
supports_images = true
max_context_length = {ctx}
auto_compact_threshold = {int(ctx * 0.78)}
{END}
'''
    if E["ACTIVATE"] == "1":
        # active_model is a root key: it must come before the first [table].
        lines = text.splitlines(keepends=True)
        first_tbl = next((i for i, l in enumerate(lines) if re.match(r"\s*\[", l)), len(lines))
        root = [l for l in lines[:first_tbl] if not re.match(r"\s*active_model\s*=", l)]
        replaced = len(root) != first_tbl
        text = "".join(root) + f'{ACT}\nactive_model = "{E["ALIAS"]}"\n\n' + "".join(lines[first_tbl:])
        if replaced:
            print("  note: replaced your previous active_model (re-run with --no-activate to keep it)")
    if text and not text.endswith("\n"):
        text += "\n"
    text += ("\n" if text.strip() else "") + block

with open(cfg_path, "w") as f:
    f.write(text)

env = open(env_path).read().splitlines() if os.path.exists(env_path) else []
env = [l for l in env if not l.startswith(KEYVAR + "=")]
if E["REMOVE"] != "1" and E["KEY"]:
    env.append(f"{KEYVAR}={E['KEY']}")
with open(env_path, "w") as f:
    f.write("\n".join(env) + ("\n" if env else ""))
os.chmod(env_path, stat.S_IRUSR | stat.S_IWUSR)

if E["REMOVE"] == "1":
    print(f"Removed the bunya-ml4 provider/model from {cfg_path} and the key from {env_path}.")
else:
    print(f"Wrote provider '{E['PROVIDER']}' -> {E['BASE']} (via {E['VIA']}, reasoning field '{E['FIELD']}')")
    print(f"      model '{E['ALIAS']}' ({E['MODEL']}, context {E['CONTEXT']}) in {cfg_path}")
    print(f"      {KEYVAR} in {env_path}" if E["KEY"] else
          f"      no key given: add {KEYVAR}=<key> to {env_path} or export it")
    print("Start Vibe with:  vibe" + ("" if E["ACTIVATE"] == "1" else f"   (then /model {E['ALIAS']})"))
PY

if (( ! REMOVE )); then
    if curl -fsS -m 5 "${BASE%/}/models" -H "Authorization: Bearer $KEY" >/dev/null 2>&1; then
        echo "Endpoint check: ${BASE%/}/models answers."
    else
        echo "Endpoint check: ${BASE%/}/models did not answer (tunnel down, server not up, or wrong key) — config written anyway."
    fi
fi
