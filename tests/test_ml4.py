"""CPU-only checks for the Mistral Large 4 launcher: no GPU, model or endpoint.

    python3 -m unittest discover -s tests -v
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest


def write(path, text):
    with open(path, "w") as f:
        f.write(text)


def read(path):
    with open(path) as f:
        return f.read()

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVE = os.path.join(REPO, "serve-ml4.sh")


def plan(**env):
    e = {k: v for k, v in os.environ.items()
         if not k.startswith(("SLURM", "ROCR_", "HIP_", "CUDA_", "GPU_DEVICE"))}
    e.update({"ML4_ENV": "/nonexistent", "ROCR_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"})
    e.update(env)
    r = subprocess.run([SERVE, "plan"], capture_output=True, text=True, env=e)
    return r.returncode, r.stdout, r.stderr


class Plan(unittest.TestCase):
    def test_mi355x_default_is_mxfp4_tp4_monokernel(self):
        rc, out, err = plan(GPU_ARCH="mi355x")
        self.assertEqual(rc, 0, err)
        self.assertIn("engine=day0 weights=mxfp4 tp=4 replicas=1 mono=1 spec=dspark(3)", out)
        self.assertIn("--monokernel", out)
        self.assertIn("--tensor-parallel-size 4", out)
        self.assertIn("--kv-cache-dtype fp8", out)
        self.assertIn("Quark-MXFP4", out)
        self.assertIn("VLLM_USE_V2_MODEL_RUNNER=1", out)
        self.assertIn("--host 0.0.0.0 --port 8000", out)

    def test_mistral_native_format_and_parsers(self):
        _, out, _ = plan(GPU_ARCH="mi355x")
        for flag in ("--tokenizer-mode mistral", "--config-format mistral", "--load-format mistral",
                     "--tool-call-parser mistral", "--reasoning-parser mistral",
                     "--enable-auto-tool-choice", "dspark"):
            self.assertIn(flag, out)

    def test_mi355x_many_agents_go_tp8_without_monokernel(self):
        rc, out, err = plan(GPU_ARCH="mi355x", AGENTS="16")
        self.assertEqual(rc, 0, err)
        self.assertIn("tp=8", out)
        self.assertIn("mono=0", out)
        self.assertNotIn("--monokernel", out)

    def test_litellm_runs_two_tp4_replicas_on_localhost(self):
        rc, out, err = plan(GPU_ARCH="mi355x", GATEWAY="litellm")
        self.assertEqual(rc, 0, err)
        self.assertIn("replicas=2", out)
        self.assertIn("GPUs 0,1,2,3 -> 127.0.0.1:8101", out)
        self.assertIn("GPUs 4,5,6,7 -> 127.0.0.1:8102", out)
        self.assertIn("# litellm: 0.0.0.0:8000", out)

    def test_four_gpu_allocation_runs_one_replica(self):
        rc, out, err = plan(GPU_ARCH="mi355x", GATEWAY="litellm", ROCR_VISIBLE_DEVICES="0,1,2,3")
        self.assertEqual(rc, 0, err)
        self.assertIn("replicas=1", out)
        self.assertIn("idle: 0", out)

    def test_mi300x_is_fp8_tp8_branch_engine(self):
        rc, out, err = plan(GPU_ARCH="mi300x")
        self.assertEqual(rc, 0, err)
        self.assertIn("arch=mi300x engine=branch weights=fp8 tp=8", out)
        self.assertIn("mono=0", out)
        self.assertIn("/scratch/opendata/models/gguf/mistral-large-4", out)
        self.assertIn("# overlay: NickLucche/vllm@28a4495de85c", out)
        self.assertNotIn("--kv-cache-dtype", out)        # bf16 KV on gfx942 until fp8 is qualified
        self.assertIn("sha256:55968af15ce4", out)

    def test_refusals(self):
        cases = [
            (dict(GPU_ARCH="mi300x", WEIGHTS="mxfp4"), "needs MI355X"),
            (dict(GPU_ARCH="mi300x", ENGINE="day0"), "gfx950 only"),
            (dict(GPU_ARCH="mi355x", WEIGHTS="fp8", PROFILE="tp4"), "needs all 8 GPUs"),
            (dict(GPU_ARCH="mi355x", REPLICAS="2"), "needs GATEWAY=litellm"),
            (dict(GPU_ARCH="mi355x", PROFILE="tp8", MONOKERNEL="1"), "needs TP4"),
            (dict(GPU_ARCH="mi355x", SPECULATIVE="eagle"), "dspark"),
            (dict(GPU_ARCH="mi355x", KV_CACHE_DTYPE="int8"), "KV_CACHE_DTYPE"),
            (dict(GPU_ARCH="mi355x", MONOKERNEL="1", KV_CACHE_DTYPE="auto"), "latent cache"),
        ]
        for env, msg in cases:
            rc, out, err = plan(**env)
            self.assertNotEqual(rc, 0, f"{env} should be refused")
            self.assertIn(msg, err, f"{env}: {err}")

    def test_speculative_none_drops_dspark(self):
        _, out, _ = plan(GPU_ARCH="mi355x", SPECULATIVE="none")
        self.assertNotIn("--speculative-config", out)
        self.assertIn("spec=none", out)

    def test_overrides_reach_argv(self):
        _, out, _ = plan(GPU_ARCH="mi355x", CONTEXT_LEN="1048576", MAX_NUM_SEQS="8", MEM_UTIL="0.8",
                         EXTRA_ENGINE_ARGS="--enable-log-requests")
        self.assertIn("--max-model-len 1048576", out)
        self.assertIn("--max-num-seqs 8", out)
        self.assertIn("--gpu-memory-utilization 0.8", out)
        self.assertTrue(out.rstrip().endswith("--enable-log-requests"))

    def test_api_key_never_printed(self):
        _, out, _ = plan(GPU_ARCH="mi355x", ML4_API_KEY="supersecret123")
        self.assertNotIn("supersecret123", out)


class LiteLLMSnippet(unittest.TestCase):
    def test_snippet_is_yaml_shaped_and_keyless(self):
        e = dict(os.environ, ML4_ENV="/nonexistent", GPU_ARCH="mi355x", ML4_SNIPPET_HOST="bun161",
                 ML4_API_KEY="supersecret123")
        out = subprocess.run([SERVE, "litellm-snippet"], capture_output=True, text=True, env=e).stdout
        self.assertIn("api_base: http://bun161:8000/v1", out)
        self.assertIn("model: hosted_vllm/mistral-large-4", out)
        self.assertIn("api_key: os.environ/ML4_BACKEND_KEY", out)
        self.assertIn('allowed_openai_params: ["reasoning_effort"]', out)
        self.assertNotIn("supersecret123", out)


class VibeSetup(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp()
        write(os.path.join(self.home, "config.toml"),
              'enable_telemetry = false\nactive_model = "devstral"\n\n[tools.bash]\ndefault_timeout = 600\n')

    def tearDown(self):
        shutil.rmtree(self.home)

    def run_setup(self, *args):
        r = subprocess.run([os.path.join(REPO, "vibe-setup.sh"), "--vibe-home", self.home, *args],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        return read(os.path.join(self.home, "config.toml"))

    def test_direct_then_litellm_then_remove(self):
        cfg = self.run_setup("--via", "direct", "--port", "8000", "--api-key", "k1")
        self.assertIn('reasoning_field_name = "reasoning"', cfg)
        self.assertIn('api_style = "openai"', cfg)
        cfg = self.run_setup("--via", "litellm", "--port", "4000", "--api-key", "k2")
        self.assertEqual(cfg.count("[[providers]]"), 1, "re-running must replace the block")
        self.assertIn('reasoning_field_name = "reasoning_content"', cfg)
        self.assertIn('api_base = "http://localhost:4000/v1"', cfg)
        # active_model stays a ROOT key (before the first table)
        root = cfg.split("[tools.bash]")[0]
        self.assertIn('active_model = "mistral-large-4-bunya"', root)
        self.assertNotIn('"devstral"', cfg)
        env = read(os.path.join(self.home, ".env"))
        self.assertEqual(env.strip(), "BUNYA_ML4_API_KEY=k2")
        self.assertEqual(os.stat(os.path.join(self.home, ".env")).st_mode & 0o777, 0o600)
        try:
            import tomllib
            d = tomllib.loads(cfg)
            self.assertEqual(d["models"][0]["thinking_levels"], ["low", "medium", "high"])
            self.assertEqual(d["tools"]["bash"]["default_timeout"], 600)
        except ImportError:
            pass
        cfg = self.run_setup("--remove")
        self.assertNotIn("bunya-ml4", cfg)
        self.assertIn("[tools.bash]", cfg)


class Affinity(unittest.TestCase):
    """litellm/ml4_affinity.py with LiteLLM stubbed out."""

    @classmethod
    def setUpClass(cls):
        import importlib.util, sys, types
        stub = types.ModuleType("litellm.integrations.custom_logger")
        stub.CustomLogger = type("CustomLogger", (), {"__init__": lambda self: None})
        for name in ("litellm", "litellm.integrations"):
            sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules["litellm.integrations.custom_logger"] = stub
        spec = importlib.util.spec_from_file_location("ml4_affinity", os.path.join(REPO, "litellm", "ml4_affinity.py"))
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def setUp(self):
        self.t = [0.0]
        self.aff = self.mod.ConversationAffinity(clock=lambda: self.t[0])
        self.deps = [{"model_info": {"id": "ml4-r0"}}, {"model_info": {"id": "ml4-r1"}}]

    def route(self, msgs, deps=None):
        return self.aff.choose(deps or self.deps, msgs)[0]["model_info"]["id"]

    @staticmethod
    def conv(n, turns=0):
        m = [{"role": "system", "content": "You are Vibe."}, {"role": "user", "content": f"task {n}"}]
        for t in range(turns):
            m += [{"role": "assistant", "content": f"a{t}", "reasoning_content": "r"},
                  {"role": "user", "content": f"more {t}"}]
        return m

    def test_new_conversations_balance_and_stay_pinned(self):
        first = [self.route(self.conv(i)) for i in range(4)]
        self.assertEqual(sorted(first), ["ml4-r0", "ml4-r0", "ml4-r1", "ml4-r1"])
        for i in range(4):
            for turns in range(1, 6):
                self.assertEqual(self.route(self.conv(i, turns)), first[i])

    def test_idle_conversations_stop_counting(self):
        self.route(self.conv(0)); self.route(self.conv(1))
        self.t[0] += self.mod.ACTIVE_S + 1
        busy = self.route(self.conv(2))
        self.assertEqual(self.route(self.conv(3)), "ml4-r0" if busy == "ml4-r1" else "ml4-r1")

    def test_unknown_conversation_in_progress_uses_stable_hash(self):
        other = self.mod.ConversationAffinity(clock=lambda: 0.0)
        a = self.route(self.conv(7, 3))
        b = other.choose(self.deps, self.conv(7, 3))[0]["model_info"]["id"]
        self.assertEqual(a, b)

    def test_unhealthy_pin_moves(self):
        rid = self.route(self.conv(0))
        survivor = [d for d in self.deps if d["model_info"]["id"] != rid]
        self.assertEqual(self.aff.choose(survivor, self.conv(0, 2))[0]["model_info"]["id"],
                         survivor[0]["model_info"]["id"])

    def test_single_deployment_untouched(self):
        self.assertEqual(self.aff.choose(self.deps[:1], self.conv(0)), self.deps[:1])

    def test_generated_config_enables_callback(self):
        cfg = read(SERVE)
        self.assertIn("callbacks: ml4_affinity.affinity", cfg)
        self.assertIn("id: ml4-r$n", cfg)


class VerifyWeights(unittest.TestCase):
    def test_complete_and_incomplete(self):
        d = tempfile.mkdtemp()
        try:
            for f in ("params.json", "tekken.json"):
                write(os.path.join(d, f), "{}")
            write(os.path.join(d, "consolidated.safetensors.index.json"),
                  json.dumps({"weight_map": {"a": "consolidated-00001-of-00002.safetensors",
                                             "b": "consolidated-00002-of-00002.safetensors"}}))
            write(os.path.join(d, "consolidated-00001-of-00002.safetensors"), "x")
            v = [os.path.join(REPO, "tools", "verify_weights.py"), d, "--quiet"]
            self.assertEqual(subprocess.run(v).returncode, 1)
            write(os.path.join(d, "consolidated-00002-of-00002.safetensors"), "x")
            self.assertEqual(subprocess.run(v).returncode, 0)
            os.makedirs(os.path.join(d, ".cache/huggingface/download"))
            write(os.path.join(d, ".cache/huggingface/download/x.incomplete"), "")
            self.assertEqual(subprocess.run(v).returncode, 1)
        finally:
            shutil.rmtree(d)


class Hygiene(unittest.TestCase):
    def test_shell_syntax(self):
        for f in ("serve-ml4.sh", "serve-ml4.sbatch", "submit-ml4.sh", "bench-ml4.sh", "vibe-setup.sh"):
            r = subprocess.run(["bash", "-n", os.path.join(REPO, f)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, f"{f}: {r.stderr}")

    def test_python_compiles(self):
        for f in ("agentstress.py", "tools/ml4check.py", "tools/verify_weights.py"):
            subprocess.run(["python3", "-m", "py_compile", os.path.join(REPO, f)], check=True)

    def test_agentstress_replays_reasoning(self):
        src = read(os.path.join(REPO, "agentstress.py"))
        self.assertIn("asst[rfield] = reasoning", src)

    def test_env_example_matches_launcher_knobs(self):
        env = read(os.path.join(REPO, "ml4-env.example"))
        for knob in ("GPU_ARCH", "WEIGHTS", "GATEWAY", "ENGINE", "AGENTS", "PROFILE", "REPLICAS",
                     "SPECULATIVE", "MONOKERNEL", "FP8_PATH", "MXFP4_PATH", "STATE_DIR", "PORT"):
            self.assertRegex(env, rf"export {knob}=", knob)

    def test_doc_links_resolve(self):
        for root, _, files in os.walk(REPO):
            if "/tests/fixtures" in root or "/.git" in root:
                continue
            for f in files:
                if not f.endswith(".md"):
                    continue
                p = os.path.join(root, f)
                for link in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", read(p)):
                    if link.startswith(("http://", "https://", "mailto:")):
                        continue
                    self.assertTrue(os.path.exists(os.path.join(os.path.dirname(p), link)),
                                    f"{p}: broken link {link}")


if __name__ == "__main__":
    unittest.main()
