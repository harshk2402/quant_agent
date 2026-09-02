"""End-to-end proof that the parsed contract is actionable.

The numerics are hand-written (no LLM yet -- that's Step 0.2), but EVERY path and
filename comes from TaskContext, nothing is hardcoded. Swap the task dir and the
same script writes to whatever that unit declares.
"""
import json, math, pathlib, sys
sys.path.insert(0, "/")
from agent.task_context import parse_task, resolve_output_dir

ctx = parse_task("/input")
src = next(i for i in ctx.inputs if i.fmt == "json").path()      # parser-resolved
out_dir = resolve_output_dir(); out_dir.mkdir(parents=True, exist_ok=True)
spec = ctx.primary_output                                        # parser-resolved
print(f"[demo] reading {src}  ->  writing {spec.filename} ({spec.fmt}) in {out_dir}")

d = json.loads(src.read_text())
mats, pars, freq = d["maturities"], d["par_rates"], int(d["coupon_freq"])
zero = {}          # maturity -> continuously-compounded zero rate

def z_at(t):
    """Zero rate at t by linear interpolation over known pillars."""
    ks = sorted(zero)
    if t in zero: return zero[t]
    if t <= ks[0]: return zero[ks[0]]
    if t >= ks[-1]: return zero[ks[-1]]
    lo = max(k for k in ks if k <= t); hi = min(k for k in ks if k >= t)
    w = (t - lo) / (hi - lo)
    return zero[lo] * (1 - w) + zero[hi] * w

for T, c in zip(mats, pars):
    n = int(round(T * freq))
    coupon = c / freq
    acc = sum(math.exp(-z_at(k / freq) * (k / freq)) for k in range(1, n))
    df_T = (1.0 - coupon * acc) / (1.0 + coupon)
    zero[T] = -math.log(df_T) / T

dfs = {T: math.exp(-zero[T] * T) for T in mats}
fwd = {}
for i, T in enumerate(mats):
    fwd[T] = zero[T] if i == 0 else (zero[T] * T - zero[mats[i-1]] * mats[i-1]) / (T - mats[i-1])

r8 = lambda v: round(v, 8)
payload = {
    "zero_rates":       {str(T): r8(zero[T]) for T in mats},
    "discount_factors": {str(T): r8(dfs[T])  for T in mats},
    "forward_rates":    {str(T): r8(fwd[T])  for T in mats},
}
spec.path_in(out_dir).write_text(json.dumps(payload, indent=2))
print("[demo] wrote", spec.path_in(out_dir))
