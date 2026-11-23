"""
Runs every algorithm one after the other and then the comparison.

    python run_all.py                 # real data in ../dataset/nthri_tea_yield.csv
    python run_all.py --quick         # fast smoke test
    python run_all.py --data FILE.csv --out FOLDER

Each algorithm can also be run on its own, e.g.  python tastfn.py
"""
import subprocess
import sys
from pathlib import Path

SCRIPTS = ["data_analysis.py", "mlr.py", "mlr_interactions.py", "stepwise_mlr.py", "decision_tree.py",
           "svr_linear.py", "svr_quadratic.py", "svr_cubic.py", "svr_gaussian.py",
           "gpr_rational_quadratic.py", "gpr_squared_exponential.py", "gpr_matern52.py",
           "gpr_exponential.py", "random_forest.py", "gradient_boosting.py", "xgboost_model.py",
           "nas_dnn.py", "tastfn.py", "compare_all.py"]

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    failed = []
    for s in SCRIPTS:
        print(f"\n{'=' * 78}\nRunning {s}\n{'=' * 78}", flush=True)
        r = subprocess.run([sys.executable, str(here / s), *sys.argv[1:]], cwd=here)
        if r.returncode != 0:
            failed.append(s)
    print("\nFinished." + (f" FAILED: {failed}" if failed else " All scripts completed successfully."))
    sys.exit(1 if failed else 0)
