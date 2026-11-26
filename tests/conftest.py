import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CLIENT_DIR = REPO_ROOT / "baml_client"


def _baml_cli() -> str | None:
    found = shutil.which("baml") or shutil.which("baml-cli")
    if found:
        return found
    local = REPO_ROOT / "node_modules" / ".bin" / "baml"
    if local.exists():
        return str(local)
    return None


def _ensure_generated_client() -> None:
    if (CLIENT_DIR / "baml_client" / "__init__.py").exists():
        return
    cli = _baml_cli()
    if cli is None:
        raise RuntimeError(
            "Generated BAML client not found and no baml CLI on PATH. "
            "Run `npm install` and then `npx baml --features beta generate` "
            "from the repository root."
        )
    subprocess.run([cli, "--features", "beta", "generate"], cwd=REPO_ROOT, check=True)


_ensure_generated_client()
sys.path.insert(0, str(CLIENT_DIR))
