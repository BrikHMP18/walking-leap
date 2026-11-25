# uv locations for running walking-leap on an Alliance Canada cluster (Rorqual).
# Source from the repository root before any uv command, on the login node and
# in jobs:  source cluster/env.sh
#
# Everything set here is regenerable with `uv sync --frozen`, so it lives on
# scratch (no backup, purged after inactivity): the virtualenv alone is ~6.5 GB
# and ~44k files, too many for /project, whose file quota the group shares.

scratch_root="${SCRATCH:-$HOME/links/scratch}"
export PATH="$HOME/.local/bin:$PATH" # uv
export UV_CACHE_DIR="$scratch_root/uv/cache"
export UV_PYTHON_INSTALL_DIR="$scratch_root/uv/python"
# Same uv-built Python on the login node and in jobs, whatever modules are loaded.
export UV_PYTHON_PREFERENCE=only-managed
export UV_PROJECT_ENVIRONMENT="$scratch_root/envs/walking-leap"
export XDG_CACHE_HOME="$scratch_root/cache" # Warp kernel cache, among others
# Credentials are tiny and must survive scratch purges: keep them in $HOME
# (the W&B key goes to ~/.netrc on its own).
export HF_TOKEN_PATH="$HOME/.cache/huggingface/token"
unset scratch_root
