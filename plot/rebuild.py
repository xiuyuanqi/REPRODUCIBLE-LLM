"""Re-read configured CSVs, verify the derived values, and export the figure."""
from pathlib import Path
import subprocess,sys

root=Path(__file__).resolve().parent
for script in ('build_data.py','verify.py','verify_panel_d.py','draw_figure.py'):
    subprocess.run([sys.executable,str(root/script)],check=True)
