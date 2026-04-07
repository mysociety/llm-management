from __future__ import annotations

from pathlib import Path
from typing import Optional

import rich
from exoscale.api.exceptions import (
    ExoscaleAPIClientException,
    ExoscaleAPIServerException,
)
from exoscale.api.v2 import Client
from pydantic import BaseModel as PydanticBaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_store import BaseModel

from .settings import CONFIG_PATH, settings


class LLMManagementError(Exception):
    """Base exception for llm-management errors."""


class DeploymentNotFoundError(LLMManagementError):
    def __init__(self, slug: str):
        super().__init__(f"Deployment {slug} not found.")


def get_client(zone: str) -> Client:
    if not settings.api_key or not settings.api_secret:
        raise LLMManagementError(
            "EXOSCALE_API_KEY and EXOSCALE_API_SECRET must be set "
            "(via environment variables or .env file)."
        )
    return Client(settings.api_key, settings.api_secret, zone=zone)


class ExoscaleDeploymentConfig(BaseModel):
    slug: str
    model: str
    gpu_type: str
    gpu_count: int
    replicas: int
    zone: str
    inference_engine_params: list[str] = []

    def _client(self) -> Client:
        return get_client(self.zone)

    def _find_deployment(self, client: Client) -> dict | None:
        """Return the deployment dict matching this slug, or None."""
        deployments = client.list_deployments().get("deployments", [])
        for d in deployments:
            if d.get("name") == self.slug:
                return d
        return None

    def is_deployed(self) -> bool:
        client = self._client()
        return self._find_deployment(client) is not None

    def model_in_zone(self) -> bool:
        client = self._client()
        models = client.list_models().get("models", [])
        return any(m.get("name") == self.model for m in models)

    def upload_model_to_zone(self):
        client = self._client()
        rich.print(f"Creating model {self.model} in zone {self.zone}...")
        kwargs: dict[str, str] = {"name": self.model}
        if settings.huggingface_token:
            kwargs["huggingface_token"] = settings.huggingface_token
        op = client.create_model(**kwargs)
        client.wait(op["id"])
        rich.print(f"Model {self.model} is ready.")

    def delete_model_from_zone(self):
        client = self._client()
        models = client.list_models().get("models", [])
        for m in models:
            if m.get("name") == self.model:
                rich.print(f"Deleting model {self.model} from zone {self.zone}...")
                op = client.delete_model(id=m["id"])
                client.wait(op["id"])
                rich.print(f"Model {self.model} deleted.")
                return

    def ensure_model(self, refresh: bool = False):
        if refresh and self.model_in_zone():
            self.delete_model_from_zone()
        if not self.model_in_zone():
            self.upload_model_to_zone()
        else:
            rich.print(f"Model {self.model} already exists in zone {self.zone}.")

    def create_deployment(self, refresh_model: bool = False):
        self.ensure_model(refresh=refresh_model)
        client = self._client()
        rich.print(f"Creating deployment {self.slug}...")
        try:
            op = client.create_deployment(
                name=self.slug,
                model={"name": self.model},
                gpu_type=self.gpu_type,
                gpu_count=self.gpu_count,
                replicas=self.replicas,
                inference_engine_parameters=self.inference_engine_params,
            )
        except ExoscaleAPIClientException as e:
            raise LLMManagementError(f"Failed to create deployment {self.slug}: {e}")
        try:
            result = client.wait(op["id"])
        except (ExoscaleAPIServerException, KeyError) as e:
            # Extract deployment ID from the error to fetch logs
            deploy_id = None
            if hasattr(e, "args") and e.args:
                # Try to get ID from the operation resource
                try:
                    resource = op.get("resource", {})
                    deploy_id = resource.get("id")
                except Exception:
                    pass
            if not deploy_id:
                # Fall back to finding it by name
                deployment = self._find_deployment(client)
                if deployment:
                    deploy_id = deployment.get("id")
            msg = f"Deployment {self.slug} failed."
            if deploy_id:
                try:
                    details = client.get_deployment(id=deploy_id)
                    state_details = details.get("state-details", "No details available")
                    msg += f"\nState details: {state_details}"
                    log_data = client.get_deployment_logs(id=deploy_id, tail=20)
                    log_lines = [
                        entry.get("message", "") for entry in log_data.get("logs", [])
                    ]
                    if log_lines:
                        msg += "\nLogs:\n" + "\n".join(
                            f"  {line}" for line in log_lines
                        )
                except Exception:
                    pass
            raise LLMManagementError(msg)
        deploy_id = result.get("resource", {}).get("id", "unknown")
        state = result.get("state", "unknown")
        rich.print(f"Deployment {self.slug} created (id: {deploy_id}, state: {state}).")

    def scale_to_zero(self):
        client = self._client()
        deployment = self._find_deployment(client)
        if deployment is None:
            raise DeploymentNotFoundError(self.slug)
        rich.print(f"Scaling {self.slug} to zero replicas...")
        op = client.scale_deployment(id=deployment["id"], replicas=0)
        client.wait(op["id"])
        rich.print(f"Deployment {self.slug} scaled to zero.")

    def resume_from_zero(self):
        client = self._client()
        deployment = self._find_deployment(client)
        if deployment is None:
            raise DeploymentNotFoundError(self.slug)
        rich.print(f"Resuming {self.slug} to {self.replicas} replica(s)...")
        op = client.scale_deployment(id=deployment["id"], replicas=self.replicas)
        client.wait(op["id"])
        rich.print(f"Deployment {self.slug} resumed.")

    def connection_info(self) -> dict:
        client = self._client()
        deployment = self._find_deployment(client)
        if deployment is None:
            raise DeploymentNotFoundError(self.slug)
        deployment_url = deployment.get("deployment-url", "")
        api_key = client.reveal_deployment_api_key(id=deployment["id"]).get("api-key")
        return {
            "url": deployment_url,
            "api_key": api_key,
        }

    def delete_deployment(self):
        client = self._client()
        deployment = self._find_deployment(client)
        if deployment is None:
            raise DeploymentNotFoundError(self.slug)
        rich.print(f"Deleting deployment {self.slug}...")
        op = client.delete_deployment(id=deployment["id"])
        client.wait(op["id"])
        rich.print(f"Deployment {self.slug} deleted.")

    def create_or_resume(self):
        """
        If the deployment does not exist, create it.
        If it exists and is scaled to zero, resume it.
        If it exists and is not scaled to zero, do nothing.
        """
        client = self._client()
        deployment = self._find_deployment(client)
        if deployment is None:
            self.create_deployment()
        elif deployment.get("replicas", 0) == 0:
            self.resume_from_zero()
        else:
            rich.print(f"Deployment {self.slug} is already running.")

    def test_deployment(self) -> bool:
        """Test the deployment by asking the LLM for the capital of France."""
        info = self.connection_info()

        class CapitalCity(PydanticBaseModel):
            country: str
            city: str

        model = OpenAIModel(
            self.model,
            provider=OpenAIProvider(
                base_url=info["url"],
                api_key=info["api_key"],
            ),
        )
        agent = Agent(
            model,
            system_prompt="Given the name of a country, return the capital city of that country. e.g. Germany, return 'Berlin'",
            output_type=CapitalCity,
        )
        result = agent.run_sync("France")
        output = result.output
        if output.city.lower() == "paris":
            return True
        raise LLMManagementError(f"Expected city='Paris' but got city='{output.city}'")


class ExoscaleConfig(BaseModel):
    """Container for all deployment configs, loaded from exoscale.toml."""

    deployment: list[ExoscaleDeploymentConfig]

    @classmethod
    def load(cls, config_path: Path = CONFIG_PATH) -> ExoscaleConfig:
        return cls.from_file(config_path)

    def get(self, slug: str) -> ExoscaleDeploymentConfig:
        for d in self.deployment:
            if d.slug == slug:
                return d
        raise LLMManagementError(f"No deployment config found with slug '{slug}'.")

    def resolve(
        self, slug: Optional[str], all_: bool
    ) -> list[ExoscaleDeploymentConfig]:
        if all_:
            return self.deployment
        if slug is None:
            raise LLMManagementError("Provide a slug or use --all.")
        return [self.get(slug)]

    def create_all(self):
        for cfg in self.deployment:
            cfg.create_deployment()

    def delete_all(self):
        for cfg in self.deployment:
            cfg.delete_deployment()

    def scale_all_to_zero(self):
        for cfg in self.deployment:
            cfg.scale_to_zero()

    def list_deployments(self):
        seen_zones: set[str] = set()
        for cfg in self.deployment:
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
