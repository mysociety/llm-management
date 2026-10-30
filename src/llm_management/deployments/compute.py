"""Single-VM deployments using prepared templates and a local SSH tunnel.

Resource names and a durable SSH key allow recovery after a manager restart.
A filesystem lock coordinates local CLI/server operations. One manager process
must own a deployment; its state directory must survive application restarts.
"""

from __future__ import annotations

import atexit
import base64
import fcntl
import ipaddress
import logging
import socket
import subprocess
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, NamedTuple, TextIO

import httpx2
from exoscale.api.v2 import Client

from ..settings import settings
from ..templates import compute
from ..templates.probes import probe, validate_health
from ..templates.state import ComputeState
from .client import get_client
from .errors import LLMManagementError
from .types import ConnectionInfo, DeploymentQueryResult

if TYPE_CHECKING:
    from .config import ComputeDeploymentConfig

logger = logging.getLogger(__name__)
_runtime_lock = threading.RLock()


class TunnelRuntime(NamedTuple):
    """
    Process and local resources owned by one SSH tunnel.
    """

    process: subprocess.Popen[bytes]
    port: int
    log_file: TextIO


_runtimes: dict[str, TunnelRuntime] = {}


def close_tunnel(name: str) -> None:
    """
    Stop a process-owned SSH tunnel and close its log handle.

    Remove its runtime entry first; force termination if graceful shutdown times out.
    This closes only local access and leaves the cloud VM running.
    """
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
    """
    Release process-owned tunnels at exit without deleting cloud resources.
    """
    for name in list(_runtimes):
        close_tunnel(name)


class ComputeDeployment:
    """
    Manage one prepared VM, its durable recovery identity and local SSH tunnel.

    Cloud operations are synchronous and belong in worker threads. Lifecycle callers
    use locked() to coordinate threads and local CLI/server processes. Durable keys
    and manifests permit restart recovery; tunnel processes exist only in memory.
    Scale-to-zero deletes the VM rather than keeping a paused deployment.
    """

    def __init__(self, config: ComputeDeploymentConfig) -> None:
        """
        Bind the recipe and establish its private state directory without starting a VM.

        Resource identity includes the zone and configured deployment name. Reject names
        unsafe for cloud resources or storage paths. One manager process must own the
        live tunnel for a deployment even though the filesystem lock coordinates changes.
        """
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
    def locked(self) -> Generator[None, None, None]:
        """
        Serialize lifecycle changes across threads and cooperating local processes.

        Hold the reentrant thread lock and filesystem lock for the entire operation.
        Release the filesystem lock on every exit, including provider or startup failures.
        """
        with self._thread_lock, (self.directory / "lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def state(self) -> ComputeState:
        """
        Load and validate a recovery manifest, or return fresh unsaved identity.

        Reject a manifest whose name or zone belongs to another deployment. Keep unknown
        JSON fields through the typed record so saving progress does not discard metadata.
        """
        if self.path.exists():
            state = ComputeState.load(self.path)
            if state.name != self.name or state.zone != self.recipe.zone:
                raise LLMManagementError(
                    "Compute manifest does not match this deployment"
                )
            return state
        return ComputeState(name=self.name, zone=self.recipe.zone)

    def instance(self, client: Client) -> dict[str, Any] | None:
        """
        Discover this deployment's VM by name, rejecting ambiguous matches.

        Return None when absent. Discovering cloud resources also recovers creations whose
        instance ID was not saved before the manager stopped.
        """
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

    def settle_creation(
        self, client: Client, state: ComputeState, instance: dict[str, Any] | None
    ) -> None:
        """
        Wait for an interrupted creation before replacing or deleting it.
        """
        if instance and instance.get("state") in {"running", "stopped"}:
            return
        if state.create_operation and not state.cleanup_complete:
            operation = client.get_operation(id=state.create_operation)
            if operation["state"] == "pending":
                compute.wait(client, {"id": state.create_operation})
            elif operation["state"] not in {"success", "failure", "timeout"}:
                raise LLMManagementError(
                    "Unrecognized VM creation state; access resources retained"
                )

    def ssh(self, client: Client, instance: dict[str, Any]) -> list[str]:
        """
        Build SSH access to an existing VM using the retained private key.

        Read the remote address from the provider and username from durable state. Missing
        keys raise a recovery error; never generate replacement access for an existing VM.
        """
        key = self.directory / "id_ed25519"
        if not key.exists():
            raise LLMManagementError(
                f"SSH key missing for existing VM; restore {key} or destroy and recreate the deployment"
            )
        details = client.get_instance(id=instance["id"])
        user = self.state().default_user
        return compute.ssh_command(
            key, self.directory / "known_hosts", f"{user}@{details['public-ip']}"
        )

    def tunnel_command(self) -> list[str]:
        """
        Describe a manual localhost:8000 tunnel to an existing VM.

        Do not start the command or ensure model readiness. Unlike connect(), this command
        uses a fixed local port and is intended for CLI connection instructions.
        """
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
        """
        Reuse a live process-owned tunnel or start one on an available local port.

        Return its loopback base URL and retain the process and log handle for cleanup.
        Process creation does not prove the SSH connection or model is ready; ensure()
        checks the process and health before returning to inference callers.
        """
        with _runtime_lock:
            runtime = _runtimes.get(self.runtime_key)
            if runtime and runtime.process.poll() is None:
                return f"http://127.0.0.1:{runtime.port}"
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
            _runtimes[self.runtime_key] = TunnelRuntime(process, port, log_file)
            return f"http://127.0.0.1:{port}"

    def health(self, base_url: str) -> bool:
        """
        Check readiness and verify the server matches the configured template recipe.

        Return False for unsuccessful HTTP responses or connection failures. A responding
        server with incompatible model metadata raises a deployment error rather than
        being accepted as the configured checkpoint.
        """
        try:
            response = httpx2.get(base_url + "/health", timeout=10)
            if not response.is_success:
                return False
            try:
                validate_health(response.json(), self.recipe)
            except (ValueError, RuntimeError) as exc:
                raise LLMManagementError(
                    f"Compute server does not match its template recipe: {exc}"
                ) from exc
            return True
        except httpx2.RequestError:
            return False

    def status(self) -> DeploymentQueryResult:
        """
        Report VM existence and readiness without creating a VM or tunnel.

        An existing VM reports zero replicas until a process-owned tunnel is live and the
        server matches its recipe and template release. Drop a stale tunnel when the VM
        is absent. Ready connection URLs include the /v1 API prefix.
        """
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
                    f"http://127.0.0.1:{runtime.port}"
                    if runtime and runtime.process.poll() is None
                    else ""
                )
            ready = bool(base_url) and self.health(base_url)
            if ready and self.state().template_id != self.recipe.resolve(client)["id"]:
                ready = False
            return DeploymentQueryResult(
                exists=True,
                replicas=int(ready),
                deployment_url=base_url + "/v1" if ready else "",
                api_key="",
            )

    def provision(self, client: Client, state: ComputeState) -> dict[str, Any]:
        """
        Create a single VM while saving enough identity for interrupted-run recovery.

        Resolve the prepared template, validate GPU availability and restrict SSH access.
        Clean stale access resources before creating a key and security group. Save the
        creation operation before waiting and the VM ID after discovery. The caller holds
        the lifecycle lock and owns cleanup if provisioning or subsequent startup fails.
        """
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
            response = httpx2.get("https://api.ipify.org", timeout=20)
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
        state.template_id = template["id"]
        state.default_user = template.get("default-user", "ubuntu")
        state.reset_creation()
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
        state.create_operation = operation["id"]
        compute.save(state, self.path)
        compute.wait(client, operation)
        instance = self.instance(client)
        if instance is None:
            raise LLMManagementError(
                "VM creation completed without a discoverable instance"
            )
        state.instance_id = instance["id"]
        compute.save(state, self.path)
        return instance

    def ensure(self) -> None:
        """
        Recover or create the VM and wait for SSH access and model readiness.

        Hold the lifecycle lock throughout startup. Settle interrupted creation before
        rediscovery, reject a changed template release, and start a stopped VM if needed.
        Reuse durable access and create a process-local tunnel to the ready server.

        On failure or interruption, close the tunnel. Delete cloud resources only when
        the VM was initially absent; leave an existing deployment available for recovery.
        If cleanup itself fails, retain recovery state and report the original failure.
        """
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
                    state.template_id
                    and state.template_id != self.recipe.resolve(client)["id"]
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
                        if _runtimes[self.runtime_key].process.poll() is not None:
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

    def delete_resources(self, client: Client, state: ComputeState) -> None:
        """
        Delete owned cloud resources only after settling interrupted creation.

        Verify VM deletion before removing its security group and SSH key. Keep access
        resources when deletion cannot be verified, allowing a later recovery attempt.
        Remove local access files and mark cleanup complete only after cloud cleanup
        succeeds. The caller must hold the deployment's lifecycle lock.
        """
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
        state.cleanup_complete = True
        compute.save(state, self.path)

    def delete(self) -> None:
        """
        Remove the VM and its access resources under the lifecycle lock.
        """
        with self.locked():
            self.delete_resources(get_client(self.recipe.zone), self.state())

    def test(self) -> None:
        """
        Ensure the VM and run recipe-defined HTTP and native-provider smoke checks.

        Retain the deployment and write diagnostic probe outputs in its state directory.
        """
        self.ensure()
        state = self.status()
        probe(state.deployment_url.removesuffix("/v1"), self.directory, self.recipe)

    def logs(self, tail: int) -> str:
        """
        Read recent remote System One journal entries without creating a VM.
        """
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

    def create_deployment(
        self, refresh_model: bool = False, *, replicas: int | None = None
    ) -> None:
        """
        Ensure one prepared VM; reject in-place model refresh requests.

        Changing cached weights requires a new template release. Preserve the shared
        backend interface while delegating startup to create_or_resume().
        """
        if refresh_model:
            raise LLMManagementError(
                "Compute models are cached in templates; prepare a new template release"
            )
        self.create_or_resume(replicas=replicas)

    def create_or_resume(self, *, replicas: int | None = None) -> None:
        """
        Ensure the VM is ready, accepting only the single-replica Compute policy.
        """
        if replicas not in (None, 1):
            raise LLMManagementError("Compute deployments support one replica.")
        self.ensure()

    def resume_from_zero(self, *, replicas: int | None = None) -> None:
        """
        Ensure a VM after scale-to-zero, recreating resources that were deleted.
        """
        self.create_or_resume(replicas=replicas)

    def scale_to_zero(self) -> None:
        """
        Delete the VM and access resources rather than pausing billable Compute.
        """
        self.delete()

    def delete_deployment(self) -> None:
        """
        Apply the same verified cleanup as Compute scale-to-zero.
        """
        self.delete()

    def query_status(self) -> DeploymentQueryResult:
        """
        Return the non-starting status snapshot through the shared backend interface.
        """
        return self.status()

    def connection_info(self) -> ConnectionInfo:
        """
        Ensure readiness and return API access plus a manual SSH tunnel command.

        The API URL uses this process's tunnel. The manual command uses localhost:8000;
        its lifetime is independent of the tunnel managed by this controller.
        """
        self.ensure()
        status = self.status()
        return ConnectionInfo(
            url=status.deployment_url,
            api_key=status.api_key,
            ssh_tunnel=self.tunnel_command(),
        )

    def is_deployed(self) -> bool:
        """
        Report cloud VM existence independently of its model readiness.
        """
        return self.status().exists

    def test_basic_deployment(self) -> None:
        """
        Run recipe smoke checks through the shared backend test interface.
        """
        self.test()

    def test_instruct_deployment(self) -> None:
        """
        Run the recipe's HTTP and structured native-provider smoke checks.
        """
        self.test()
