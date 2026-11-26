import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BAML_SRC_DIR = REPO_ROOT / "baml_src"
CLIENT_DIR = REPO_ROOT / "baml_client"


def _baml_cli() -> str | None:
    local_bin = REPO_ROOT / "node_modules" / ".bin"
    local = shutil.which("baml", path=str(local_bin)) or shutil.which(
        "baml-cli", path=str(local_bin)
    )
    if local:
        return local
    return shutil.which("baml") or shutil.which("baml-cli")


def _client_is_up_to_date() -> bool:
    init_py = CLIENT_DIR / "__init__.py"
    inlined_py = CLIENT_DIR / "inlinedbaml.py"
    if not (init_py.exists() and inlined_py.exists()):
        return False
    baml_files = list(BAML_SRC_DIR.rglob("*.baml"))
    if not baml_files:
        return True
    latest_src_mtime = max(
        BAML_SRC_DIR.stat().st_mtime,
        *(p.stat().st_mtime for p in baml_files),
    )
    return inlined_py.stat().st_mtime >= latest_src_mtime


def _ensure_generated_client() -> None:
    if _client_is_up_to_date():
        return
    cli = _baml_cli()
    if cli is None:
        raise RuntimeError(
            "Generated BAML client is missing or out of date and no baml CLI was found. "
            "Run `npm install` and then `npx baml --features beta generate` "
            "from the repository root."
        )
    subprocess.run([cli, "--features", "beta", "generate"], cwd=REPO_ROOT, check=True)


_ensure_generated_client()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
