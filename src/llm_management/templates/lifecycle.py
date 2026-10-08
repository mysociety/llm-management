"""Prepare, register and verify named templates with recoverable cleanup."""

import base64
import json
import shlex
import signal
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

from exoscale.api.v2 import Client

from ..models import LLMManagementError, get_client
from . import compute
from .config import TemplateRecipe


def cloud_init(recipe: TemplateRecipe) -> str:
    values = {
        "SYSTEMONE_IMAGE": recipe.image,
        "SYSTEMONE_BACKEND": recipe.backend,
        "SYSTEMONE_MODEL": recipe.model,
        "SYSTEMONE_REVISION": recipe.revision,
        "SYSTEMONE_MAX_LENGTH": str(recipe.max_length),
        "SYSTEMONE_PORT": str(recipe.port),
        "NVIDIA_DRIVER_PACKAGE": recipe.nvidia_driver_package,
    }
    assignments = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in values.items()
    )
    data = {
        "write_files": [
            {
                "path": "/opt/llm-template/bootstrap.sh",
                "content": Path(__file__).with_name("bootstrap.sh").read_text(),
            }
        ],
        "runcmd": [
            assignments
            + " bash /opt/llm-template/bootstrap.sh > /var/log/llm-template-bootstrap.log 2>&1"
        ],
    }
    return base64.b64encode(("#cloud-config\n" + json.dumps(data)).encode()).decode()


def prepare_template(
    client: Client,
    recipe: TemplateRecipe,
    state: dict[str, Any],
    path: Path,
    ssh: list[str],
    instance: dict,
    template: dict,
) -> None:
    compute.log("Sanitizing builder and preparing an offline boot service")
    command = """sudo bash -s <<'SANITIZE'
set -euo pipefail
systemctl stop systemone.service
if docker inspect systemone >/dev/null 2>&1; then
    docker stop systemone
    docker rm systemone
fi
rm -f /opt/llm-template/failed
rm -f /root/.ssh/authorized_keys /home/ubuntu/.ssh/authorized_keys
rm -f /root/.bash_history /home/ubuntu/.bash_history
rm -f /etc/ssh/ssh_host_*
cloud-init clean --logs --machine-id
sync
SANITIZE"""
    result = subprocess.run(ssh + [command], capture_output=True, timeout=120)
    (path.parent / "sanitize.log").write_bytes(result.stdout + result.stderr)
    result.check_returncode()
    compute.wait(client, client.stop_instance(id=instance["id"]))
    compute.log("Creating and exporting the sanitized snapshot")
    operation = client.create_snapshot(id=instance["id"])
    state["snapshot_operation"] = operation["id"]
    compute.save(state, path)
    result = compute.wait(client, operation)
    state["snapshot_id"] = result["reference"]["id"]
    compute.save(state, path)
    compute.wait(client, client.export_snapshot(id=state["snapshot_id"]))
    snapshot = client.get_snapshot(id=state["snapshot_id"])
    state["template_name"] = recipe.name
    compute.save(state, path)
    if recipe.find(client) is not None:
        raise LLMManagementError(
            f"Template {recipe.name!r} already exists; use a new versioned name"
        )
    compute.log("Registering the prepared template in " + state["zone"])
    operation = client.register_template(
        name=state["template_name"],
        description=recipe.description,
        url=snapshot["export"]["presigned-url"],
        checksum=snapshot["export"]["md5sum"],
        size=snapshot["size"],
        default_user=template.get("default-user", "ubuntu"),
        boot_mode=template.get("boot-mode", "legacy"),
        ssh_key_enabled=True,
        password_enabled=False,
    )
    state["template_operation"] = operation["id"]
    compute.save(state, path)
    result = compute.wait(client, operation)
    state["template_id"] = result["reference"]["id"]
    compute.save(state, path)
    compute.log("Prepared template registered: " + state["template_id"])


def cleanup(client: Client, state: dict[str, Any], path: Path) -> None:
    if not state["name"].startswith("llm-template-"):
        raise ValueError("Not an llm-management template manifest")
    # Recover snapshots even if interruption happened before their ID was saved.
    instance_ids = {state["instance_id"]} if state.get("instance_id") else set()
    instance_ids.update(
        i["id"]
        for i in client.list_instances().get("instances", [])
        if i.get("name") == state["name"]
    )
    for snapshot in client.list_snapshots().get("snapshots", []):
        if (snapshot.get("instance") or {}).get("id") in instance_ids:
            compute.log("Deleting temporary snapshot " + snapshot["id"])
            compute.wait(client, client.delete_snapshot(id=snapshot["id"]))
    compute.cleanup(client, state, path)


def execute(
    client: Client,
    recipe: TemplateRecipe,
    path: Path,
    *,
    template_id: str | None = None,
) -> dict[str, Any]:
    state = {
        "name": f"llm-template-{uuid.uuid4().hex[:12]}",
        "zone": recipe.zone,
        "image": recipe.image,
        "source_template_id": template_id,
        "recipe": recipe.model_dump(mode="json"),
    }
    compute.save(state, path)
    try:
        compute.run(
            client,
            recipe,
            state,
            path,
            user_data=cloud_init(recipe)
            if not template_id
            else base64.b64encode(b"#cloud-config\n{}").decode(),
            template_id=template_id,
            on_ready=(
                None
                if template_id
                else lambda ssh, instance, template: prepare_template(
                    client, recipe, state, path, ssh, instance, template
                )
            ),
        )
    except BaseException as exc:
        state["error_type"] = type(exc).__name__
        compute.save(state, path)
        raise
    finally:
        old_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
        old_term = signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            cleanup(client, state, path)
        except Exception:
            compute.log(
                f"Cleanup incomplete; run llm-management templates cleanup {path}"
            )
            raise
        finally:
            signal.signal(signal.SIGINT, old_int)
            signal.signal(signal.SIGTERM, old_term)
    return state


def build(recipe: TemplateRecipe, runs: int = 2, *, test_only: bool = False) -> Path:
    """Create a recipe once, or test an existing template; retain only the template."""
    if runs < 1:
        raise LLMManagementError("At least one fresh VM test is required")
    try:
        from pydantic_ai.models.system_one import SystemOneModel  # noqa: F401
    except ImportError as exc:
        raise LLMManagementError(
            "Native System One support is required; run poetry install"
        ) from exc
    client = get_client(recipe.zone)
    template = recipe.resolve(client) if test_only else recipe.find(client)
    if template is not None and not test_only:
        raise LLMManagementError(
            f"Template {recipe.name!r} already exists; use templates test or a new versioned name"
        )
    directory = Path(tempfile.mkdtemp(prefix="llm-template-"))
    compute.log("State and logs: " + str(directory))
    report = {"recipe": recipe.model_dump(mode="json"), "benchmarks": []}
    report_path = directory / "report.json"
    compute.save(report, report_path)
    previous = signal.signal(
        signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    try:
        if not test_only:
            builder = directory / "builder"
            builder.mkdir()
            report["builder"] = execute(client, recipe, builder / "state.json")
            template_id = report["builder"]["template_id"]
        else:
            assert template is not None  # test_only resolves an existing template.
            template_id = template["id"]
        report["template_id"] = template_id
        compute.save(report, report_path)
        for number in range(runs):
            run_dir = directory / f"benchmark-{number + 1}"
            run_dir.mkdir()
            report["benchmarks"].append(
                execute(client, recipe, run_dir / "state.json", template_id=template_id)
            )
            compute.save(report, report_path)
    finally:
        signal.signal(signal.SIGTERM, previous)
    compute.log("Report: " + str(report_path))
    return report_path
