"""Build every compute pipeline of the engine's shader modules on this GPU and report the ones
its compiler rejects, with the message. Each kernel builds in a process of its own: after one
internal compiler error, Metal fails every later build in the same process. Exit status 1 if
any kernel fails.

(Apple's paravirtual GPU, GitHub's M1 runners, crashed on long chains of 3 × 3 math over
9-float arrays; climb.wgsl keeps that math on mat3x3f.)

    python tests/check_shaders.py
"""
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SH = Path(__file__).resolve().parent.parent / "shaders"
read = lambda n: (SH / n).read_text(encoding="utf-8")

MODULES = {
    "climb": lambda: read("climb.wgsl"),
    "smi": lambda: read("smi.wgsl") + "\n" + read("copula.wgsl"),
    "sweep": lambda: read("sweep.wgsl") + "\n" + read("copula.wgsl"),
    "ncc": lambda: read("ncc.wgsl"),
    "ffd": lambda: "alias Texel = f32;\n" + read("ffd.wgsl"),
}


def entries(code):
    return re.findall(r"@compute\s+@workgroup_size\([^)]*\)\s*fn\s+(\w+)", code)


def one(module, entry):
    """Build one kernel in this process; print its compiler message if it fails."""
    import zcmir

    reg = zcmir.Registrar()
    msg = reg._compile_wgsl(MODULES[module](), entry)
    if msg is not None:
        lines = [l.strip() for l in msg.strip().splitlines() if l.strip()]
        print(lines[-1] if lines else msg)
        sys.exit(1)


def main():
    import zcmir

    reg = zcmir.Registrar()
    print("adapter:", reg.adapter, flush=True)
    reg.close()
    jobs = []
    for name, code in MODULES.items():
        for entry in entries(code()):
            jobs.append((name, entry))

    def run(job):
        r = subprocess.run([sys.executable, __file__, "--one", *job], capture_output=True, text=True, timeout=300)
        return job, r.returncode, (r.stdout.strip().splitlines() or [r.stderr.strip()[-300:]])[-1]

    with ThreadPoolExecutor(4) as ex:
        results = list(ex.map(run, jobs))
    bad = 0
    for (name, entry), rc, msg in results:
        if rc != 0:
            bad += 1
            print(f"FAIL {name}/{entry}: {msg}")
    for name in MODULES:
        n = sum(1 for (m, _), rc, _ in results if m == name)
        print(f"{name}: {n} kernels checked")
    print("shaders OK" if bad == 0 else f"{bad} kernels failed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--one":
        one(sys.argv[2], sys.argv[3])
    else:
        main()
