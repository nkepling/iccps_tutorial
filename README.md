# CPS-IoT Week Tutorial — Non-Stationary Decision Making with NS-Gym

A 90-minute hands-on workshop on modeling and solving non-stationary MDPs with [NS-Gym](https://nsgym.io). Designed to run from your laptop in a sandboxed Python environment — nothing is installed globally.

## Install

Three steps.

### 1. Install `uv`

`uv` is a fast Python package manager from Astral. It manages the sandboxed environment that runs the notebook.

**macOS / Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Other install methods: <https://docs.astral.sh/uv/getting-started/installation/>.

After installing, restart your shell (or `source ~/.zshrc` / `source ~/.bashrc`) so the `uv` command is on your `PATH`.

### 2. Clone this repo

```bash
git clone https://github.com/nkepling/iccps_tutorial.git
cd iccps_tutorial
```

### 3. Run the notebook

```bash
uvx marimo run --sandbox tutorial.py
```

That's it. `uv` reads the pinned dependency list at the top of `tutorial.py` (PEP 723 inline script metadata), builds an isolated venv, and starts marimo. A browser tab opens with the tutorial.

> **First run takes ~30–60 s** while `uv` downloads numba, ns-gym, gymnasium, etc. into the sandbox. Subsequent runs use the cached venv and start instantly. Nothing pollutes your global Python.



## More

- NS-Gym docs: <https://nsgym.io>
- NS-Gym source: <https://github.com/scope-lab-vu/ns_gym>
- AAMAS 2026 competition: <https://nsgym.io/aamas2026_competition.html>
