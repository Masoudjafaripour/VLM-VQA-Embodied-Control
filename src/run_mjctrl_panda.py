import os
import subprocess
import sys

repo_root = os.path.dirname(os.path.dirname(__file__))
mjctrl_dir = os.path.join(repo_root, "external", "mjctrl")

# subprocess.run(
#     [sys.executable, "diffik_nullspace.py"],
#     cwd=mjctrl_dir,
#     check=True,
# )

subprocess.run(
    [sys.executable, "diffik.py"],
    cwd=mjctrl_dir,
    check=True,
)
