"""Repair MLflow model artifacts written from Windows paths.

The six completed training runs are reused: no model training is performed.
The script rewrites MLmodel artifact-relative paths to POSIX separators, uses explicit UTF-8 when reading the
stored manifest (important under Python -I on Windows), and configures UTF-8 stdio plus the CPU thread policy.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path, PurePosixPath

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

def normalize_mlmodel(model_file: Path) -> tuple[bool, list[str], dict]:
    metadata = yaml.safe_load(model_file.read_text(encoding="utf-8"))
    changed: list[str] = []
    for flavor_name, flavor in metadata.get("flavors", {}).items():
        if not isinstance(flavor, dict):
            continue
        artifacts = flavor.get("artifacts", {})
        if not isinstance(artifacts, dict):
            continue
        for name, artifact in artifacts.items():
            if not isinstance(artifact, dict):
                continue
            path = artifact.get("path")
            if isinstance(path, str) and "\\" in path:
                fixed = path.replace("\\", "/")
                while "//" in fixed:
                    fixed = fixed.replace("//", "/")
                artifact["path"] = fixed
                changed.append(f"{flavor_name}.artifacts.{name}: {path!r} -> {fixed!r}")
    if changed:
        model_file.write_text(
            yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
            newline="\n",
        )
    return bool(changed), changed, metadata


def patch_model_code(source_file: Path) -> bool:
    source = source_file.read_text(encoding="utf-8")
    if "class AdapterModel" not in source or "def load_context(self, context):" not in source:
        return False

    # Python -I ignores PYTHONUTF8. On Windows, read_text() without an explicit
    # encoding can decode this UTF-8 JSON manifest using a legacy code page,
    # corrupting its Russian system prompt. Keep the artifact self-contained.
    manifest_read_old = 'self.manifest = json.loads((root / "manifest.json").read_text())'
    manifest_read_new = 'self.manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))'
    if manifest_read_old in source:
        source = source.replace(manifest_read_old, manifest_read_new, 1)
    elif manifest_read_new not in source:
        raise RuntimeError(f"{source_file}: не найдена строка чтения manifest.json")

    helper = '''\n\ndef _configure_utf8_stdio():
    """Keep MLflow/Hugging Face subprocess output decodable on Windows."""
    import sys

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="backslashreplace")
            except (OSError, ValueError):
                pass

'''
    if "def _configure_utf8_stdio():" not in source:
        source = source.replace("class AdapterModel", helper + "class AdapterModel", 1)

    load_context = "    def load_context(self, context):\n"
    if load_context not in source:
        return False
    if load_context + "        _configure_utf8_stdio()\n" not in source:
        source = source.replace(
            load_context,
            load_context + "        _configure_utf8_stdio()\n",
            1,
        )

    # Match the CPU intra-op thread policy used by src/train.py. Without this,
    # a fresh isolated Python process can use a different CPU kernel/threading
    # configuration and greedy generation may diverge for a very short run.
    thread_line = "        torch.set_num_threads(min(4, os.cpu_count() or 1))\n"
    if "torch.set_num_threads(min(4, os.cpu_count() or 1))" not in source:
        torch_import = "        import torch\n"
        pos = source.find(torch_import, source.find(load_context))
        if pos < 0:
            raise RuntimeError(f"{source_file}: не найден import torch внутри load_context")
        end = pos + len(torch_import)
        source = source[:end] + thread_line + source[end:]

    compile(source, str(source_file), "exec")
    source_file.write_text(source, encoding="utf-8", newline="\n")
    return True


def upload_file(client: MlflowClient, run_id: str, local_file: Path, artifact_path: str) -> None:
    client.log_artifact(run_id, str(local_file), artifact_path=artifact_path or None)


def get_model_code_path(metadata: dict) -> str | None:
    pyfunc = metadata.get("flavors", {}).get("python_function", {})
    path = pyfunc.get("model_code_path")
    if not isinstance(path, str) or not path.strip():
        return None
    # MLflow 2.22 resolves models-from-code files by basename under the model root.
    return PurePosixPath(path.replace("\\", "/")).name


def repair_one(client, run_id: str) -> None:
    from mlflow.exceptions import MlflowException

    print(f"run {run_id}: checking model artifacts", flush=True)
    with tempfile.TemporaryDirectory(prefix="hw67-repair-") as tmp:
        root = Path(tmp)
        mlmodel_file = Path(client.download_artifacts(run_id, "model/MLmodel", str(root)))
        path_changed, path_notes, metadata = normalize_mlmodel(mlmodel_file)
        if path_changed:
            upload_file(client, run_id, mlmodel_file, "model")
            print("  normalized: " + "; ".join(path_notes), flush=True)
        else:
            print("  MLmodel artifact paths already use POSIX separators", flush=True)

        code_name = get_model_code_path(metadata)
        if not code_name:
            raise RuntimeError(f"{run_id}: MLmodel не содержит python_function.model_code_path")
        code_dir = root / "code"
        code_dir.mkdir(exist_ok=True)
        try:
            code_file = Path(client.download_artifacts(run_id, f"model/{code_name}", str(code_dir)))
        except MlflowException:
            # Fallback for MLflow builds that place model code in a subdirectory.
            entries = client.list_artifacts(run_id, "model")
            candidates = [x.path for x in entries if not x.is_dir and x.path.endswith(".py")]
            code_file = None
            for artifact in candidates:
                try:
                    candidate = Path(client.download_artifacts(run_id, artifact, str(code_dir)))
                except MlflowException:
                    continue
                if "class AdapterModel" in candidate.read_text(encoding="utf-8"):
                    code_file = candidate
                    break
            if code_file is None:
                raise RuntimeError(f"{run_id}: не найден Python-код AdapterModel в артефакте model")
            code_name = code_file.name

        if patch_model_code(code_file):
            # model_code_path is stored at model/<basename> by MLflow 2.22.x.
            upload_file(client, run_id, code_file, "model")
            print(f"  patched UTF-8 manifest decoding / stdio / CPU policy in model/{code_name}", flush=True)
        else:
            source = code_file.read_text(encoding="utf-8")
            if "_configure_utf8_stdio()" not in source:
                raise RuntimeError(f"{run_id}: не удалось безопасно настроить UTF-8 в {code_name}")
            if 'read_text(encoding="utf-8")' not in source or 'manifest.json' not in source:
                raise RuntimeError(f"{run_id}: чтение manifest.json не закреплено на UTF-8")
            print(f"  UTF-8 manifest decoding already configured in model/{code_name}", flush=True)

        # Verify the remote MLmodel was actually overwritten, rather than trusting
        # the successful return from log_artifact alone.
        verify_dir = root / "verify"
        verify_dir.mkdir(exist_ok=True)
        verify_file = Path(client.download_artifacts(run_id, "model/MLmodel", str(verify_dir)))
        verify_meta = yaml.safe_load(verify_file.read_text(encoding="utf-8"))
        for flavor_name, flavor in verify_meta.get("flavors", {}).items():
            for name, artifact in (flavor.get("artifacts", {}) or {}).items():
                path = artifact.get("path") if isinstance(artifact, dict) else None
                if isinstance(path, str) and "\\" in path:
                    raise RuntimeError(
                        f"{run_id}: remote MLmodel still contains a Windows separator in "
                        f"{flavor_name}.artifacts.{name}.path={path!r}"
                    )
        code_verify_dir = root / "verify_code"
        code_verify_dir.mkdir(exist_ok=True)
        code_verify = Path(client.download_artifacts(run_id, f"model/{code_name}", str(code_verify_dir)))
        verified_source = code_verify.read_text(encoding="utf-8")
        expected_manifest_read = 'self.manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))'
        if expected_manifest_read not in verified_source:
            raise RuntimeError(f"{run_id}: сервер не сохранил явное UTF-8-чтение manifest.json")
        client.set_tag(run_id, "artifact_compatibility_patch", "windows-posix-paths+utf8-stdio+cpu-policy+manifest-utf8-v3")


def main() -> int:
    from mlflow import MlflowClient
    from src.common import connect, load_params

    params = load_params()
    connect(params)
    summary_path = ROOT / "metrics" / "sweep.json"
    if not summary_path.is_file():
        raise SystemExit("Нет metrics/sweep.json; сначала выполните make sweep")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    run_ids = list(dict.fromkeys(row["run_id"] for row in summary.get("rows", []) if row.get("run_id")))
    if len(run_ids) < 6:
        raise SystemExit(f"В metrics/sweep.json найдено только {len(run_ids)} runs, нужно минимум 6")

    client = MlflowClient()
    for run_id in run_ids:
        repair_one(client, run_id)
    print(f"Готово: исправлены метаданные/код {len(run_ids)} runs; веса и результаты обучения не менялись.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
