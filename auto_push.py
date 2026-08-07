import subprocess
from pathlib import Path

repo = Path(r"C:\Users\quang\Downloads\GNN\HIN-GNN")

commands = [
    ["git", "add", "."],
    ["git", "commit", "-m", "Automatic update"],
    ["git", "push"]
]

for cmd in commands:
    subprocess.run(cmd, cwd=repo)