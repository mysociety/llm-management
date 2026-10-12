"""
CLI presentation of managed deployments and prepared VMs.
"""

from __future__ import annotations

import rich

from .backends import backend_for
from .client import get_client
from .config import ComputeDeploymentConfig, ExoscaleDeployments


def print_deployments(config: ExoscaleDeployments) -> None:
    for cfg in config.deployment:
        if isinstance(cfg, ComputeDeploymentConfig):
            status = backend_for(cfg).query_status()
            rich.print(
                f"  {cfg.deployment_name:<20} backend=exoscale_compute exists={status.exists} ready={bool(status.replicas)}"
            )
    seen_zones: set[str] = set()
    for cfg in config.deployment:
        if isinstance(cfg, ComputeDeploymentConfig):
            continue
        if cfg.zone in seen_zones:
            continue
        seen_zones.add(cfg.zone)
        client = get_client(cfg.zone)
        deployments = client.list_deployments().get("deployments", [])
        if not deployments:
            rich.print(f"No deployments in zone {cfg.zone}.")
            continue
        rich.print(f"\nZone: {cfg.zone}")
        for d in deployments:
            rich.print(
                f"  {d.get('name', 'unnamed'):<20} "
                f"state={d.get('state', '?'):<10} "
                f"replicas={d.get('replicas', '?')}"
            )
