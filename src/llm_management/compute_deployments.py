"""Single-VM deployments using prepared templates and a local SSH tunnel.

Resource names and a durable SSH key allow recovery after a manager restart.
A filesystem lock coordinates local CLI/server operations. One manager process
must own a deployment; its state directory must survive application restarts.
"""

from __future__ import annotations

import atexit
import base64
from contextlib import contextmanager
import fcntl
import ipaddress
import json
import logging
import socket
import subprocess
import threading
import time
from typing import TYPE_CHECKING

import httpx

from .models import DeploymentQueryResult, LLMManagementError, get_client
from .settings import settings
from .templates import compute
from .templates.probes import probe, validate_health

if TYPE_CHECKING:
    from .models import ComputeDeploymentConfig

logger = logging.getLogger(__name__)
_runtime_lock = threading.RLock()
_runtimes: dict[str, tuple[subprocess.Popen, int, object]] = {}


def close_tunnel(name: str) -> None:
    with _runtime_lock:
        runtime = _runtimes.pop(name, None)
        if runtime:
            process, _, log_file = runtime
            try:
                process.terminate()
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            finally:
                log_file.close()


@atexit.register
def close_all_tunnels() -> None:
    for name in list(_runtimes):
        close_tunnel(name)


class ComputeDeployment:
    def __init__(self, config: ComputeDeploymentConfig):
        self.config = config
        self.recipe = config.recipe
        self.name = f"llm-compute-{self.recipe.zone}-{config.deployment_name}"
        # Roles appear in both resource names and local storage paths.
        if any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for c in self.name
        ):
            raise LLMManagementError(
                "Compute deployment names and server roles must use letters, digits, underscores or hyphens"
            )
        self.directory = settings.compute_state_dir / self.recipe.zone / self.name
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = self.directory / "state.json"
        self.runtime_key = str(self.directory.resolve())
        self._thread_lock = threading.RLock()

    @contextmanager
    def locked(self):
        with self._thread_lock, (self.directory / "lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def state(self) -> dict:
        if self.path.exists():
            state = json.loads(self.path.read_text())
            if state.get("name") != self.name or state.get("zone") != self.recipe.zone:
                raise LLMManagementError(
                    "Compute manifest does not match this deployment"
                )
            return state
        return {"name": self.name, "zone": self.recipe.zone}

    def instance(self, client) -> dict | None:
        matches = [
            i
            for i in client.list_instances().get("instances", [])
            if i.get("name") == self.name
        ]
        if len(matches) > 1:
            raise LLMManagementError(
                f"Multiple VMs named {self.name}; resolve the ambiguity before continuing"
            )
        return matches[0] if matches else None

    def settle_creation(self, client, state: dict, instance: dict | None) -> None:
        """Wait for an interrupted creation before replacing or deleting it."""
        if instance and instance.get("state") in {"running", "stopped"}:
            return
        if state.get("create_operation") and not state.get("cleanup_complete"):
            operation = client.get_operation(id=state["create_operation"])
            if operation["state"] == "pending":
                compute.wait(client, {"id": state["create_operation"]})
            elif operation["state"] not in {"success", "failure", "timeout"}:
                raise LLMManagementError(
                    "Unrecognized VM creation state; access resources retained"
                )

    def ssh(self, client, instance: dict) -> list[str]:
        key = self.directory / "id_ed25519"
        if not key.exists():
            raise LLMManagementError(
                f"SSH key missing for existing VM; restore {key} or destroy and recreate the deployment"
            )
        details = client.get_instance(id=instance["id"])
        user = self.state().get("default_user", "ubuntu")
        return compute.ssh_command(
            key, self.directory / "known_hosts", f"{user}@{details['public-ip']}"
        )

    def tunnel_command(self) -> list[str]:
        client = get_client(self.recipe.zone)
        instance = self.instance(client)
        if instance is None:
            raise LLMManagementError("Compute VM is absent")
        ssh = self.ssh(client, instance)
        return ssh[:-1] + [
            "-N",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
            "-L",
            f"127.0.0.1:8000:127.0.0.1:{self.recipe.port}",
            ssh[-1],
        ]

    def connect(self, ssh: list[str]) -> str:
        with _runtime_lock:
            runtime = _runtimes.get(self.runtime_key)
            if runtime and runtime[0].poll() is None:
                return f"http://127.0.0.1:{runtime[1]}"
            close_tunnel(self.runtime_key)
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            command = ssh[:-1] + [
                "-N",
                "-o",
                "ExitOnForwardFailure=yes",
                "-o",
                "ServerAliveInterval=15",
                "-o",
                "ServerAliveCountMax=3",
                "-L",
                f"127.0.0.1:{port}:127.0.0.1:{self.recipe.port}",
                ssh[-1],
            ]
            log_file = (self.directory / "ssh.log").open("a")
            try:
                process = subprocess.Popen(command, stdout=log_file, stderr=log_file)
            except BaseException:
                log_file.close()
                raise
            _runtimes[self.runtime_key] = process, port, log_file
            return f"http://127.0.0.1:{port}"

    def health(self, base_url: str) -> bool:
        try:
            response = httpx.get(base_url + "/health", timeout=10)
            if not response.is_success:
                return False
            try:
                validate_health(response.json(), self.recipe)
            except (ValueError, RuntimeError) as exc:
                raise LLMManagementError(
                    f"Compute server does not match its template recipe: {exc}"
                ) from exc
            return True
        except httpx.RequestError:
            return False

    def status(self) -> DeploymentQueryResult:
        with self.locked():
            client = get_client(self.recipe.zone)
            instance = self.instance(client)
            if instance is None:
                close_tunnel(self.runtime_key)
                return DeploymentQueryResult(
                    exists=False, replicas=0, deployment_url="", api_key=""
                )
            with _runtime_lock:
                runtime = _runtimes.get(self.runtime_key)
                base_url = (
                    f"http://127.0.0.1:{runtime[1]}"
                    if runtime and runtime[0].poll() is None
                    else ""
                )
            ready = bool(base_url) and self.health(base_url)
            if (
                ready
                and self.state().get("template_id") != self.recipe.resolve(client)["id"]
            ):
                ready = False
            return DeploymentQueryResult(
                exists=True,
                replicas=int(ready),
                deployment_url=base_url + "/v1" if ready else "",
                api_key="",
            )

    def provision(self, client, state: dict) -> dict:
        template = self.recipe.resolve(client)
        types = [
            t
            for t in client.list_instance_types().get("instance-types", [])
            if f"{t['family']}.{t['size']}" == self.recipe.instance_type
            and self.recipe.zone in t.get("zones", [])
            and t.get("authorized")
        ]
        if len(types) != 1:
            raise LLMManagementError(
                f"Instance type {self.recipe.instance_type} is unavailable in {self.recipe.zone}"
            )
        cidr = self.recipe.ssh_cidr
        if not cidr:
            response = httpx.get("https://api.ipify.org", timeout=20)
            response.raise_for_status()
            cidr = str(ipaddress.IPv4Address(response.text.strip())) + "/32"
        cidr = str(ipaddress.IPv4Network(cidr))
        # Recover access resources left by an interrupted build with no VM.
        self.delete_resources(client, state)
        key = self.directory / "id_ed25519"
        if not key.exists():
            subprocess.run(
                ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)],
                check=True,
            )
            key.chmod(0o600)
        state.update(
            template_id=template["id"],
            default_user=template.get("default-user", "ubuntu"),
        )
        state.pop("cleanup_complete", None)
        state.pop("create_operation", None)
        state.pop("instance_id", None)
        compute.save(state, self.path)
        keys = client.list_ssh_keys().get("ssh-keys", [])
        if not any(k.get("name") == self.name for k in keys):
            compute.wait(
                client,
                client.register_ssh_key(
                    name=self.name, public_key=key.with_suffix(".pub").read_text()
                ),
            )
        groups = [
            g
            for g in client.list_security_groups().get("security-groups", [])
            if g.get("name") == self.name
        ]
        if not groups:
            compute.wait(
                client,
                client.create_security_group(
                    name=self.name, description="llm-management Compute deployment"
                ),
            )
            groups = [
                g
                for g in client.list_security_groups().get("security-groups", [])
                if g.get("name") == self.name
            ]
            compute.wait(
                client,
                client.add_rule_to_security_group(
                    id=groups[0]["id"],
                    flow_direction="ingress",
                    protocol="tcp",
                    start_port=22,
                    end_port=22,
                    network=cidr,
                ),
            )
        if len(groups) != 1:
            raise LLMManagementError("Ambiguous Compute security group")
        operation = client.create_instance(
            name=self.name,
            instance_type={"id": types[0]["id"]},
            template={"id": template["id"]},
            disk_size=self.recipe.disk_size_gib,
            ssh_key={"name": self.name},
            security_groups=[{"id": groups[0]["id"]}],
            user_data=base64.b64encode(b"#cloud-config\n{}").decode(),
        )
        state["create_operation"] = operation["id"]
        compute.save(state, self.path)
        compute.wait(client, operation)
        instance = self.instance(client)
        if instance is None:
            raise LLMManagementError(
                "VM creation completed without a discoverable instance"
            )
        state["instance_id"] = instance["id"]
        compute.save(state, self.path)
        return instance

    def ensure(self) -> None:
        with self.locked():
            client = get_client(self.recipe.zone)
            state = self.state()
            instance = self.instance(client)
            created = instance is None
            try:
                self.settle_creation(client, state, instance)
                instance = self.instance(client)
                if instance is None:
                    instance = self.provision(client, state)
                if (
                    state.get("template_id")
                    and state["template_id"] != self.recipe.resolve(client)["id"]
                ):
                    raise LLMManagementError(
                        "Existing VM uses a different template release; destroy it before switching releases"
                    )
                if instance.get("state") == "stopped":
                    compute.wait(client, client.start_instance(id=instance["id"]))
                ssh = self.ssh(client, instance)
                deadline = time.monotonic() + self.recipe.startup_timeout
                while time.monotonic() < deadline:
                    result = subprocess.run(
                        ssh + ["true"], capture_output=True, timeout=20
                    )
                    if result.returncode == 0:
                        break
                    time.sleep(self.recipe.poll_interval)
                else:
                    raise LLMManagementError("Compute SSH startup timed out")
                base_url = self.connect(ssh)
                while time.monotonic() < deadline:
                    with _runtime_lock:
                        if _runtimes[self.runtime_key][0].poll() is not None:
                            raise LLMManagementError(
                                f"SSH tunnel exited; see {self.directory / 'ssh.log'}"
                            )
                    if self.health(base_url):
                        return
                    time.sleep(self.recipe.poll_interval)
                raise LLMManagementError("Compute model startup timed out")
            except BaseException:
                close_tunnel(self.runtime_key)
                if created:
                    try:
                        self.delete_resources(client, state)
                    except Exception:
                        logger.exception(
                            "Compute cleanup failed; run destroy %s to retry",
                            self.config.slug,
                        )
                raise

    def delete_resources(self, client, state: dict) -> None:
        close_tunnel(self.runtime_key)
        instance = self.instance(client)
        self.settle_creation(client, state, instance)
        instance = self.instance(client)
        if instance:
            compute.wait(client, client.delete_instance(id=instance["id"]))
        if self.instance(client) is not None:
            raise LLMManagementError(
                "VM deletion could not be verified; access resources retained"
            )
        for group in client.list_security_groups().get("security-groups", []):
            if group.get("name") == self.name:
                compute.wait(client, client.delete_security_group(id=group["id"]))
        for key in client.list_ssh_keys().get("ssh-keys", []):
            if key.get("name") == self.name:
                compute.wait(client, client.delete_ssh_key(name=self.name))
        if any(
            g.get("name") == self.name
            for g in client.list_security_groups().get("security-groups", [])
        ) or any(
            k.get("name") == self.name
            for k in client.list_ssh_keys().get("ssh-keys", [])
        ):
            raise LLMManagementError(
                "Compute access-resource deletion could not be verified"
            )
        for name in ("id_ed25519", "id_ed25519.pub", "known_hosts"):
            (self.directory / name).unlink(missing_ok=True)
        state["cleanup_complete"] = True
        compute.save(state, self.path)

    def delete(self) -> None:
        with self.locked():
            self.delete_resources(get_client(self.recipe.zone), self.state())

    def test(self) -> None:
        self.ensure()
        state = self.status()
        probe(state.deployment_url.removesuffix("/v1"), self.directory, self.recipe)

    def logs(self, tail: int) -> str:
        client = get_client(self.recipe.zone)
        instance = self.instance(client)
        if instance is None:
            raise LLMManagementError("Compute VM is absent")
        result = subprocess.run(
            self.ssh(client, instance)
            + [f"sudo journalctl -u systemone.service -n {int(tail)} --no-pager"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        return result.stdout
