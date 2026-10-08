"""Temporary Compute provisioning and recovery for template builds/tests."""

import ipaddress
import json
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from exoscale.api.v2 import Client

import httpx

from .probes import probe, validate_health
from .config import TemplateRecipe


def log(message: str) -> None:
    print(message, flush=True)


def wait(client: Client, operation: dict[str, Any]) -> dict[str, Any]:
    return client.wait(operation["id"], max_wait_time=1200)


def save(state: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def cleanup(client: Client, state: dict[str, Any], path: Path) -> None:
    """Discover by the unique run name, including creations interrupted mid-wait."""
    name = state["name"]
    # Refuse broad cleanup or a manifest describing an unrelated resource.
    if not name.startswith("llm-template-"):
        raise ValueError("Not an llm-management template manifest")
    instances = [
        i for i in client.list_instances().get("instances", []) if i.get("name") == name
    ]
    for instance in instances:
        log(f"Deleting temporary VM {instance['id']}")
        wait(client, client.delete_instance(id=instance["id"]))
    remaining = [
        i for i in client.list_instances().get("instances", []) if i.get("name") == name
    ]
    if remaining:
        raise RuntimeError(
            f"VM deletion was not verified; run templates cleanup {path}"
        )
    state["vm_deleted"] = True
    save(state, path)
    groups = [
        g
        for g in client.list_security_groups().get("security-groups", [])
        if g.get("name") == name
    ]
    for group in groups:
        wait(client, client.delete_security_group(id=group["id"]))
    keys = [
        k for k in client.list_ssh_keys().get("ssh-keys", []) if k.get("name") == name
    ]
    for key in keys:
        wait(client, client.delete_ssh_key(name=key["name"]))
    if any(
        g.get("name") == name
        for g in client.list_security_groups().get("security-groups", [])
    ) or any(k.get("name") == name for k in client.list_ssh_keys().get("ssh-keys", [])):
        raise RuntimeError(
            f"Access resource deletion was not verified; run templates cleanup {path}"
        )
    state["cleanup_complete"] = True
    save(state, path)
    log("Cleanup verified: VM, SSH key and security group deleted")


def ssh_command(key: Path, known_hosts: Path, host: str) -> list[str]:
    return [
        "ssh",
        "-i",
        str(key),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        host,
    ]


def run(
    client: Client,
    args: TemplateRecipe,
    state: dict[str, Any],
    path: Path,
    *,
    user_data: str,
    template_id: str | None = None,
    on_ready: Callable[[list[str], dict, dict], None] | None = None,
) -> None:
    name = state["name"]
    types = client.list_instance_types().get("instance-types", [])
    instance_type = next(
        (
            t
            for t in types
            if f"{t['family']}.{t['size']}" == args.instance_type
            and args.zone in t.get("zones", [])
        ),
        None,
    )
    if not instance_type or not instance_type.get("authorized"):
        raise RuntimeError(
            f"GPU Compute type {args.instance_type} is unavailable or unauthorized in {args.zone}"
        )
    if template_id:
        template = client.get_template(id=template_id)
    else:
        templates = client.list_templates(visibility="public").get("templates", [])
        matches = [t for t in templates if t["name"] == args.base_template_name]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one public OS template named {args.base_template_name!r}, found {len(matches)}"
            )
        template = matches[0]
    cidr = args.ssh_cidr
    if not cidr:
        response = httpx.get("https://api.ipify.org", timeout=20)
        response.raise_for_status()
        cidr = f"{ipaddress.IPv4Address(response.text.strip())}/32"
    cidr = str(ipaddress.IPv4Network(cidr))
    key = path.parent / "id_ed25519"
    subprocess.run(
        ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)], check=True
    )
    key.chmod(0o600)
    log(f"Registering SSH key and security group; SSH source {cidr}")
    wait(
        client,
        client.register_ssh_key(
            name=name, public_key=key.with_suffix(".pub").read_text()
        ),
    )
    wait(
        client,
        client.create_security_group(
            name=name, description="Temporary template build/test"
        ),
    )
    group = next(
        g for g in client.list_security_groups()["security-groups"] if g["name"] == name
    )
    wait(
        client,
        client.add_rule_to_security_group(
            id=group["id"],
            flow_direction="ingress",
            protocol="tcp",
            start_port=22,
            end_port=22,
            network=cidr,
        ),
    )
    log(f"Creating {args.instance_type} VM in {args.zone}")
    started = time.monotonic()
    operation = client.create_instance(
        name=name,
        instance_type={"id": instance_type["id"]},
        template={"id": template["id"]},
        disk_size=args.disk_size_gib,
        ssh_key={"name": name},
        security_groups=[{"id": group["id"]}],
        user_data=user_data,
    )
    state["create_operation"] = operation["id"]
    save(state, path)
    wait(client, operation)
    state["provision_seconds"] = round(time.monotonic() - started, 3)
    instance = next(
        i for i in client.list_instances()["instances"] if i["name"] == name
    )
    state["instance_id"] = instance["id"]
    save(state, path)
    details = client.get_instance(id=instance["id"])
    host = f"{template.get('default-user', 'ubuntu')}@{details['public-ip']}"
    ssh = ssh_command(key, path.parent / "known_hosts", host)
    deadline = time.monotonic() + args.startup_timeout
    log("Waiting for SSH and model loading")
    while time.monotonic() < deadline:
        result = subprocess.run(ssh + ["true"], capture_output=True, timeout=20)
        if result.returncode == 0:
            state["ssh_seconds"] = round(time.monotonic() - started, 3)
            break
        time.sleep(10)
    else:
        raise TimeoutError("VM SSH did not become ready")
    # Both HTTP and the SSH tunnel listen on loopback.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    tunnel_command = ssh[:-1] + [
        "-N",
        "-o",
        "ExitOnForwardFailure=yes",
        "-L",
        f"127.0.0.1:{port}:127.0.0.1:{args.port}",
        host,
    ]
    with (path.parent / "ssh.log").open("w") as ssh_log:
        tunnel = subprocess.Popen(tunnel_command, stdout=ssh_log, stderr=ssh_log)
        try:
            base_url = f"http://127.0.0.1:{port}"
            while time.monotonic() < deadline:
                if tunnel.poll() is not None:
                    raise RuntimeError("SSH tunnel exited; see ssh.log")
                try:
                    response = httpx.get(base_url + "/health", timeout=10)
                    if response.is_success:
                        validate_health(response.json(), args)
                        log(json.dumps(response.json()))
                        state["health_seconds"] = round(time.monotonic() - started, 3)
                        probe(base_url, path.parent, args)
                        state["smoke_complete_seconds"] = round(
                            time.monotonic() - started, 3
                        )
                        state["inference_passed"] = True
                        save(state, path)
                        if on_ready:
                            on_ready(ssh, instance, template)
                        return
                except httpx.RequestError:
                    pass
                status = subprocess.run(
                    ssh
                    + [
                        "if test -f /opt/llm-template/failed; then printf failed; else sudo docker inspect -f '{{.State.Status}}' systemone 2>/dev/null || true; fi"
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if status.stdout.strip() in {"failed", "exited", "dead"}:
                    raise RuntimeError(
                        "Remote bootstrap or model startup failed; see remote.log"
                    )
                log("Waiting for model startup...")
                time.sleep(args.poll_interval)
            raise TimeoutError("model startup timed out; see remote.log")
        finally:
            tunnel.terminate()
            tunnel.wait(timeout=15)
            result = subprocess.run(
                ssh
                + [
                    "sudo tail -n 100 /var/log/llm-template-bootstrap.log; sudo docker logs --tail 100 systemone"
                ],
                capture_output=True,
                timeout=30,
            )
            (path.parent / "remote.log").write_bytes(result.stdout + result.stderr)
