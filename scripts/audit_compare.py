"""Compare cached exit codes (and block reasons) with the old per-chart e2e run.
usage: audit_compare.py CACHE OLD_RUN_DIR"""
import json, pickle, sys, collections
from pathlib import Path
sys.path.insert(0, "/home/cky10/tactilegen_runs/wt_new"); sys.dont_write_bytecode = True
cache, old = Path(sys.argv[1]), Path(sys.argv[2])
n = same = 0; diffs = []; secs = []; sa_new = sa_old = 0
for f in sorted(cache.glob("*.pkl")):
    r = pickle.loads(f.read_bytes()); cid = r["id"]
    o = json.load(open(old / "charts" / cid / "result.json"))
    ostd = (old / "charts" / cid / "stdout.txt").read_text()
    n += 1; secs.append(r["seconds"])
    ok = str(o["exit"]) == str(r["exit"]) and ("series_ambiguous" in ostd) == ("series_ambiguous" in r["stdout"])
    sa_new += "series_ambiguous" in r["stdout"]; sa_old += "series_ambiguous" in ostd
    same += ok
    if not ok: diffs.append((cid, o["exit"], r["exit"]))
secs.sort()
print(f"n={n} identical exit+series_ambiguous: {same}; series_ambiguous new/old {sa_new}/{sa_old}")
print("exit new:", collections.Counter(str(pickle.loads(f.read_bytes())['exit']) for f in cache.glob('*.pkl')))
print("diffs:", diffs[:20])
print(f"s/chart (in-worker) median {secs[len(secs)//2]} p95 {secs[int(.95*len(secs))-1]} sum {sum(secs):.0f}")
