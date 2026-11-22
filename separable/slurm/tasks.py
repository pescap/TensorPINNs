import sys

LEGACY = "--norm 0 --scale 0 --hardaux 0 --weights legacy --lr 1e-4 --const-lr 1 --fixed 1 --adam 10000 --lbfgs 0 --width 50 --tag legacy"


def pilot():
    for seed in range(3):
        s = f"--seed {seed}"
        for k in range(1, 7):
            yield f"--problem sep --arch lr --k {k} --rank 8 {s}"
        for k in range(1, 7):
            yield f"--problem sep --arch mo --k {k} {s}"
            yield f"--problem sep --arch mo --k {k} --balance 1 --tag bal {s}"
        for k in range(1, 5):
            yield f"--problem sep --arch mo --k {k} {LEGACY} {s}"
            yield f"--problem sep --arch mo --k {k} --scale 0 --weights legacy --tag hardaux_only {s}"
            yield f"--problem sep --arch mo --k {k} --hardaux 0 --tag scale_only {s}"
        for k in range(1, 5):
            yield f"--problem sep --arch v --k {k} {s}"
        for amp in (1.0, 2.0):
            a = f"--problem logn --amp {amp}"
            for k in range(2, 7):
                for rank in (8, 16, 32):
                    yield f"{a} --arch lr --k {k} --rank {rank} --tag a{amp}_r{rank} {s}"
            for k in range(2, 5):
                yield f"{a} --arch mo --k {k} --tag a{amp} {s}"
                yield f"{a} --arch mo --k {k} --balance 1 --tag a{amp}_bal {s}"
                yield f"{a} --arch v --k {k} --tag a{amp} {s}"


def stage2():
    for seed in range(3):
        s = f"--seed {seed}"
        for k in range(2, 7):
            yield f"--problem sep --arch mo --k {k} --split 1 --tag split {s}"
            yield f"--problem sep --arch mo --k {k} --split 1 --sequential 1 --tag seq {s}"
            yield f"--problem sep --arch mo --k {k} --split 1 --sequential 1 --adam {20000 * k} --lbfgs {5000 * k} --tag seqfull {s}"
        for k in range(2, 5):
            yield f"--problem logn --arch mo --k {k} --split 1 --sequential 1 --tag seq {s}"
        for prob in ("sep", "logn"):
            for k in (2, 3, 4):
                for wbc in (1, 10, 100):
                    yield f"--problem {prob} --arch lr --k {k} --bc soft --wbc {wbc} --rank 32 --tag w{wbc} {s}"
                    yield f"--problem {prob} --arch mo --k {k} --bc soft --wbc {wbc} --tag w{wbc} {s}"
                    if k < 4:
                        yield f"--problem {prob} --arch v --k {k} --bc soft --wbc {wbc} --tag w{wbc} {s}"
        for rhs in ("sep", "gauss", "exp"):
            h = f"--problem helm --rhs {rhs} --k 2"
            for rank in (16, 32, 64):
                yield f"{h} --arch lr --rank {rank} --tag {rhs}_r{rank} {s}"
            yield f"{h} --arch lr --rank 32 --bc soft --tag {rhs}_soft {s}"
            yield f"{h} --arch mo --tag {rhs} {s}"
            yield f"{h} --arch v --tag {rhs} {s}"
        for rhs in ("gauss", "exp"):
            h = f"--problem heat --rhs {rhs} --k 2"
            for rank in (16, 32, 64):
                yield f"{h} --arch lr --rank {rank} --tag {rhs}_r{rank} {s}"
            yield f"{h} --arch mo --tag {rhs} {s}"
            yield f"{h} --arch v --tag {rhs} {s}"


def hole():
    for seed in range(3):
        s = f"--seed {seed}"
        h = "--problem hole --modes 4 --mesh 96"
        for k in (2, 3, 4):
            for rank in (16, 32):
                yield f"{h} --arch lr --k {k} --rank {rank} --tag r{rank} {s}"
            for wbc in (1, 10, 100):
                yield f"{h} --arch lr --k {k} --rank 32 --bc soft --wbc {wbc} --tag w{wbc} {s}"
        for cutoff in (0.15, 0.6):
            yield f"{h} --arch lr --k 2 --rank 32 --cutoff {cutoff} --tag c{cutoff} {s}"
        for k in (2, 3):
            yield f"{h} --arch mo --k {k} {s}"
        yield f"{h} --arch v --k 2 {s}"
        yield f"{h} --arch v --k 2 --bc soft --tag w1 {s}"


def peak():
    for seed in range(3):
        s = f"--seed {seed} --nl 16000"
        for amp in (1.0, 2.0):
            a = f"--problem logn --amp {amp}"
            for k in range(2, 7):
                for rank in (16, 32):
                    yield f"{a} --arch lr --k {k} --rank {rank} --peak 1 --tag a{amp}_r{rank}_peak {s}"
            for k in (3, 4):
                yield f"{a} --arch mo --k {k} --peak 1 --tag a{amp}_peak {s}"
            yield f"{a} --arch v --k 3 --peak 1 --tag a{amp}_peak {s}"
        for k in (4, 6, 8, 10):
            b = f"--problem logn --modes 4 --tuples 1 --arch lr --k {k}"
            for rank in (16, 32):
                yield f"{b} --rank {rank} --peak 1 --tag m4_r{rank}_peak {s}"
            yield f"{b} --rank 16 --tag m4_r16 {s}"


def box(v=False):
    for seed in range(3):
        for d in (2, 3):
            for c2 in (0.0, -4 * 3.141592653589793**2):
                for corr in (0.05, 0.01):
                    b = f"--problem box --dim {d} --k 2 --c2 {c2} --corr {corr} --seed {seed}"
                    t = f"d{d}_c{'h' if c2 else 'p'}_l{corr}"
                    if v:
                        yield f"{b} --arch v --tag {t}"
                        continue
                    for rank in (16, 32, 64, 128):
                        yield f"{b} --arch lr --rank {rank} --tag {t}_r{rank}"
                    yield f"{b} --arch mo --tag {t}"


def boxv():
    return box(True)


HIGHD = [("gauss", 4, 0.1), ("gauss", 6, 0.15), ("gauss", 8, 0.2), ("exp", 2, 0.1), ("exp", 3, 0.1), ("exp", 4, 0.2)]


def highd(v=False, width=128):
    for seed in range(3):
        for rhs, d, corr in HIGHD:
            b = f"--problem box --rhs {rhs} --dim {d} --k 2 --corr {corr} --width {width} --seed {seed}"
            t = f"{rhs}_d{d}_w{width}"
            if v:
                if d <= 4:
                    yield f"{b} --arch v --tag {t}"
                continue
            for rank in (64, 128):
                yield f"{b} --arch lr --rank {rank} --grid 1000 --tag {t}_r{rank}_g"
            yield f"{b} --arch mo --tag {t}"


def highdv():
    return highd(True)


def ball():
    for seed in range(3):
        b = f"--problem ball --modes 4 --width 128 --seed {seed}"
        for rank in (32, 64):
            yield f"{b} --k 2 --arch lr --rank {rank} --grid 400 --tag r{rank}_g"
        yield f"{b} --k 2 --arch mo --tag base"
        yield f"{b} --k 2 --arch v --tag base"
        for k in (3, 4):
            for rank in (32, 64):
                yield f"{b} --k {k} --arch lr --rank {rank} --peak 1 --nl 16000 --tag r{rank}_peak"
            yield f"{b} --k {k} --arch mo --peak 1 --nl 16000 --tag peak"


def paper():
    for seed in range(3):
        s = f"--seed {seed} --tag paper"
        for k in range(1, 5):
            yield f"--problem sep --arch lr --k {k} --rank 8 {s}"
            yield f"--problem sep --arch mo --k {k} {s}"
            yield f"--problem sep --arch v --k {k} {s}"
    yield from paperkernels()
    b = "--problem ball --modes 4 --width 128 --seed 0 --tag paper"
    yield f"{b} --k 2 --arch lr --rank 64 --grid 1000"
    for k in (3, 4):
        yield f"{b} --k {k} --arch lr --rank 64 --peak 1 --nl 16000"


def paperball():
    b = "--problem ball --modes 4 --width 128 --seed 0 --tag paper"
    yield f"{b} --k 2 --arch mo"
    yield f"{b} --k 2 --arch v"
    for k in (3, 4):
        yield f"{b} --k {k} --arch mo --peak 1 --nl 16000"


def paperkernels():
    for prob, rhss in (("helm", ("sep", "gauss", "exp")), ("heat", ("gauss", "exp"))):
        for rhs in rhss:
            s = f"--problem {prob} --rhs {rhs} --k 2 --seed 0 --tag paper_{rhs}"
            yield f"{s} --arch lr --rank 64 --grid 1000"
            yield f"{s} --arch mo"
            yield f"{s} --arch v"


def peakis():
    for seed in range(3):
        for k in (8, 10):
            yield f"--problem logn --modes 4 --tuples 1 --arch lr --k {k} --rank 32 --peak 1 --nl 16000 --tag m4_r32_peak --seed {seed}"


def ballw():
    for seed in range(3):
        b = f"--problem ball --modes 4 --width 128 --seed {seed} --arch lr --rank 64"
        yield f"{b} --k 2 --grid 1000 --tag r64_g"
        for k in (3, 4):
            yield f"{b} --k {k} --peak 1 --nl 16000 --tag r64_peak"


if __name__ == "__main__":
    print("\n".join(globals()[sys.argv[1]]()))
