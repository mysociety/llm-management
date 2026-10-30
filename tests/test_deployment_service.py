"""Application lifecycle contracts exercised without importing the HTTP server."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llm_management.deployments.cache import DeploymentCache
from llm_management.deployments.errors import DeploymentServiceError
from llm_management.deployments.service import DeploymentService
from llm_management.deployments.types import DeploymentState
from llm_management.inference.resources import LocalResource, ResourceRegistry


def test_activity_lease_prevents_scaling_and_releases_after_failure():
    cache = DeploymentCache()
    state = DeploymentState(slug="model", exists=True, replicas=1)
    cache.set(state)
    config = SimpleNamespace(scale_to_zero=Mock())
    service = DeploymentService(
        SimpleNamespace(get=lambda slug: config), cache, ResourceRegistry()
    )
    cache.status.ensure = lambda backend: state
    service.backends["model"] = SimpleNamespace(
        config=config, scale_to_zero=config.scale_to_zero
    )

    async def run():
        with pytest.raises(ValueError, match="caller failed"):
            with service.track_requests():
                await service.ensure_running("model")
                await service.ensure_running("model")
                assert service.activity.get("model").requests_in_flight == 1
                with pytest.raises(DeploymentServiceError) as failure:
                    await service.scale_to_zero("model")
                assert failure.value.status_code == 409
                raise ValueError("caller failed")

    asyncio.run(run())
    assert service.activity.get("model").requests_in_flight == 0
    config.scale_to_zero.assert_not_called()


def test_explicit_local_warmup_needs_no_remote_lookup():
    catalog = Mock()
    registry = ResourceRegistry()
    loader = Mock(return_value="loaded")
    resource = registry.register(LocalResource("tokenizer", loader, Mock()))
    service = DeploymentService(catalog, DeploymentCache(), registry)

    assert asyncio.run(service.warmup_resource("tokenizer")) is None
    assert resource.status().ready
    loader.assert_called_once()
    catalog.get.assert_not_called()
