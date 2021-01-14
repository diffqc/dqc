import torch
import pprofile
from dqc.qccalc.ks import KS
from dqc.system.mol import Mol

def run_ks_forward(moldesc, basis="6-311++G**", xc="lda_x", grid="sg3"):
    # run a simple KS energy calculation
    mol = Mol(moldesc, basis=basis, grid=grid)
    qc = KS(mol, xc=xc).run()
    ene = qc.energy()
    return ene

def cmd():
    # change the command below to the commands you want to profile
    run_ks_forward("O 0 0 -2; C 0 0 2", xc="lda_x", grid="sg3")

if __name__ == "__main__":
    import time

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiler", action="store_const", default=False, const=True)
    args = parser.parse_args()

    if args.profiler:
        prof = pprofile.Profile()
        with prof:
            cmd()
        prof.print_stats()
    else:
        t0 = time.time()
        cmd()
        print("Elapsed time: %fs" % (time.time() - t0))
