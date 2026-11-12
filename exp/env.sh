# Manual activation of samurai-env (micromamba is not on PATH in the sandbox).
export E=$HOME/micromamba/envs/samurai-env
export PATH=$E/bin:$PATH
export CONDA_PREFIX=$E
export W=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
