#!/usr/bin/env python3
"""Check that a Mistral Large 4 checkpoint directory is complete.

Usage: verify_weights.py <path> [<repo> <revision>] [--offline] [--quiet] [--write-manifest]

Offline: the Mistral-native files are present (params.json, tekken.json,
consolidated.safetensors.index.json) and every shard the index names exists
with a non-zero size, and no half-written download is lying around.
Online (default when repo + revision are given): additionally, every file in
the Hub tree at that exact revision exists locally with the same size.

--write-manifest records what is there (repo, revision, sizes) in
BUNYA-MANIFEST.json beside the weights, for the next user of the shared store.

Exit 0 complete, 1 incomplete, 2 bad usage. Runs on the host's python3 (3.9+).
"""
import datetime
import getpass
import json
import os
import sys
import urllib.request

REQUIRED = ("params.json", "tekken.json", "consolidated.safetensors.index.json")
# Hub files that are not needed to serve and may legitimately differ.
OPTIONAL = {"README.md", ".gitattributes", "BUNYA-MANIFEST.json"}


def main(argv):
    flags = {a for a in argv if a.startswith("--")}
    pos = [a for a in argv if not a.startswith("--")]
    if not pos or len(pos) not in (1, 3):
        print(__doc__.strip().splitlines()[2], file=sys.stderr)
        return 2
    path = pos[0]
    repo, rev = (pos[1], pos[2]) if len(pos) == 3 else (None, None)
    quiet = "--quiet" in flags
    offline = "--offline" in flags or repo is None

    def say(msg):
        if not quiet:
            print("  " + msg)

    problems = []
    if not os.path.isdir(path):
        say(f"MISSING directory {path}")
        return 1
    for f in REQUIRED:
        if not os.path.isfile(os.path.join(path, f)):
            problems.append(f"missing {f}")
    shards = set()
    idx = os.path.join(path, "consolidated.safetensors.index.json")
    if os.path.isfile(idx):
        try:
            with open(idx) as fh:
                shards = set(json.load(fh)["weight_map"].values())
        except Exception as e:  # noqa: BLE001
            problems.append(f"unreadable index: {e}")
    missing = sorted(s for s in shards
                     if not os.path.isfile(os.path.join(path, s))
                     or os.path.getsize(os.path.join(path, s)) == 0)
    if missing:
        problems.append(f"{len(missing)} of {len(shards)} shards missing/empty (first: {missing[0]})")
    dl = os.path.join(path, ".cache", "huggingface", "download")
    if os.path.isdir(dl):
        partial = [f for _, _, fs in os.walk(dl) for f in fs if f.endswith(".incomplete")]
        if partial:
            problems.append(f"{len(partial)} unfinished download(s) in .cache/huggingface (download still running or interrupted)")

    remote = None
    if not offline and not problems:
        url = f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true&limit=1000"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "bunya-ml4-verify"})
            tok = os.environ.get("HF_TOKEN")
            if tok:
                req.add_header("Authorization", f"Bearer {tok}")
            with urllib.request.urlopen(req, timeout=60) as r:
                remote = [x for x in json.load(r) if x.get("type") == "file"]
        except Exception as e:  # noqa: BLE001
            say(f"(could not reach the Hub for a size check: {e}; offline checks only)")
        if remote:
            bad = []
            for x in remote:
                if x["path"] in OPTIONAL:
                    continue
                lp = os.path.join(path, x["path"])
                ls = os.path.getsize(lp) if os.path.isfile(lp) else None
                if ls != x["size"]:
                    bad.append(f"{x['path']} (local {ls}, hub {x['size']})")
            if bad:
                problems.append(f"{len(bad)} file(s) differ from {repo}@{rev[:8]}: {bad[0]}")

    total = sum(os.path.getsize(os.path.join(path, s)) for s in shards
                if os.path.isfile(os.path.join(path, s)))
    if problems:
        for p in problems:
            say("FAIL " + p)
        return 1
    say(f"OK   {len(shards)} shards, {total / 1e9:.1f} GB"
        + ("" if remote is None else f", sizes match {repo}@{rev[:8]}"))

    if "--write-manifest" in flags and repo:
        man = {
            "repo": repo, "revision": rev,
            "verified": datetime.datetime.now().isoformat(timespec="seconds"),
            "by": getpass.getuser(), "tool": "serve-ml4.sh (AMD_MI355X_Bunya_LLM_tools_mistral-large-4)",
            "shards": len(shards), "shard_bytes": total,
            "files": sorted(os.listdir(path)),
        }
        try:
            mp = os.path.join(path, "BUNYA-MANIFEST.json")
            with open(mp, "w") as fh:
                json.dump(man, fh, indent=1)
            os.chmod(mp, 0o644)
            say(f"wrote {mp}")
        except OSError as e:
            say(f"(manifest not written: {e})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
