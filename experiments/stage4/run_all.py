"""Run Stage 4: windowed in-match PPR with warm starts, then its figures."""

from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
EXPERIMENTS = ["e1_dynamic_ppr.py", "e2_figures.py"]


if __name__ == "__main__":
    for script in EXPERIMENTS:
        print(f"Running {script}", flush=True)
        subprocess.run([sys.executable, str(HERE / script), *sys.argv[1:]], check=True)
