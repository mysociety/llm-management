"""
Managed Exoscale inference deployment operations.
"""

from __future__ import annotations

import time
from typing import Any

import rich
from exoscale.api.exceptions import (
    ExoscaleAPIClientException,
    ExoscaleAPIServerException,
)
from exoscale.api.v2 import Client
from pydantic import BaseModel as PydanticBaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from ..settings import settings
from .client import get_client
from .config import ExoscaleDeploymentConfig
from .errors import DeploymentNotFoundError, LLMManagementError
from .types import ConnectionInfo, DeploymentQueryResult


class ManagedDeployment:
    """
    Operate an Exoscale-managed inference deployment using validated configuration.

    Cloud calls are synchronous; async callers run them in worker threads. Scaling to
    zero keeps deployment identity and the imported model. Catalog validation and
    controller construction perform no cloud operations.
    """

    def __init__(self, config: ExoscaleDeploymentConfig) -> None:
        """
        Retain configuration without contacting the provider.
        """
        self.config = config

    def client(self) -> Client:
        """
        Construct an authenticated management client for the configured zone.
        """
        return get_client(self.config.zone)

    def find_deployment(self, client: Client) -> dict[str, Any] | None:
        """
        Return the deployment dict matching this deployment_name, or None.
        """
        deployments = client.list_deployments().get("deployments", [])
        for d in deployments:
            if d.get("name") == self.config.deployment_name:
                return d
        return None

    def is_deployed(self) -> bool:
        """
        Report whether deployment identity exists, including zero-replica deployments.
        """
        client = self.client()
        return self.find_deployment(client) is not None

    def model_in_zone(self) -> bool:
        """
        Check whether the repository has been imported into this zone.
        """
        client = self.client()
        models = client.list_models().get("models", [])
        return any(m.get("name") == self.config.model for m in models)

    def upload_model_to_zone(self) -> None:
        """
        Import the model repository using configured Hugging Face credentials.

        Wait for the import operation before allowing deployment creation.
        """
        client = self.client()
        rich.print(f"Creating model {self.config.model} in zone {self.config.zone}...")
        kwargs: dict[str, str] = {"name": self.config.model}
        if settings.huggingface_token:
            kwargs["huggingface_token"] = settings.huggingface_token
        op = client.create_model(**kwargs)
        client.wait(op["id"])
        rich.print(f"Model {self.config.model} is ready.")

    def delete_model_from_zone(self) -> None:
        """
        Delete the imported repository from this zone, or return when absent.
        """
        client = self.client()
        models = client.list_models().get("models", [])
        for m in models:
            if m.get("name") == self.config.model:
                rich.print(
                    f"Deleting model {self.config.model} from zone {self.config.zone}..."
                )
                op = client.delete_model(id=m["id"])
                client.wait(op["id"])
                rich.print(f"Model {self.config.model} deleted.")
                return

    def ensure_model(self, refresh: bool = False) -> None:
        """
        Import an absent model, optionally deleting and reimporting an existing one.
        """
        if refresh and self.model_in_zone():
            self.delete_model_from_zone()
        if not self.model_in_zone():
            self.upload_model_to_zone()
        else:
            rich.print(
                f"Model {self.config.model} already exists in zone {self.config.zone}."
            )

    def create_deployment(
        self, refresh_model: bool = False, *, replicas: int | None = None
    ) -> None:
        """
        Ensure the imported model and wait for a new managed deployment.

        Use configured replicas unless an explicit target is supplied. On selected
        provider startup failures, collect deployment state and recent logs when possible
        and raise a management error. This operation does not deduplicate existing
        identity; callers wanting that behavior use create_or_resume().
        """
        self.ensure_model(refresh=refresh_model)
        client = self.client()
        _t0 = time.monotonic()
        rich.print(f"Creating deployment {self.config.deployment_name}...")
        try:
            op = client.create_deployment(
                name=self.config.deployment_name,
                model={"name": self.config.model},
                gpu_type=self.config.gpu_type,
                gpu_count=self.config.gpu_count,
                replicas=self.config.replicas if replicas is None else replicas,
                inference_engine_parameters=self.config.inference_engine_params,
            )
        except ExoscaleAPIClientException as e:
            raise LLMManagementError(
                f"Failed to create deployment {self.config.deployment_name}: {e}"
            )
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
                deployment = self.find_deployment(client)
                if deployment:
                    deploy_id = deployment.get("id")
            msg = f"Deployment {self.config.deployment_name} failed."
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
        _elapsed = time.monotonic() - _t0
        rich.print(
            f"Deployment {self.config.deployment_name} created (id: {deploy_id}, state: {state}) in {_elapsed:.1f}s."
        )

    def scale_to_zero(self) -> None:
        """
        Wait for zero replicas while retaining deployment identity and its model.

        Raise DeploymentNotFoundError if the deployment does not exist.
        """
        client = self.client()
        deployment = self.find_deployment(client)
        if deployment is None:
            raise DeploymentNotFoundError(self.config.slug)
        rich.print(f"Scaling {self.config.deployment_name} to zero replicas...")
        op = client.scale_deployment(id=deployment["id"], replicas=0)
        client.wait(op["id"])
        rich.print(f"Deployment {self.config.deployment_name} scaled to zero.")

    def resume_from_zero(self, *, replicas: int | None = None) -> None:
        """
        Scale an existing deployment to its configured or explicit replica count.

        This also resizes running deployments. Wait for the provider operation; an absent
        deployment raises DeploymentNotFoundError rather than being created here.
        """
        client = self.client()
        deployment = self.find_deployment(client)
        if deployment is None:
            raise DeploymentNotFoundError(self.config.slug)
        target = self.config.replicas if replicas is None else replicas
        _t0 = time.monotonic()
        rich.print(f"Resuming {self.config.deployment_name} to {target} replica(s)...")
        op = client.scale_deployment(id=deployment["id"], replicas=target)
        client.wait(op["id"])
        _elapsed = time.monotonic() - _t0
        rich.print(
            f"Deployment {self.config.deployment_name} resumed in {_elapsed:.1f}s."
        )

    def connection_info(self) -> ConnectionInfo:
        """
        Read the endpoint and attach the configured inference credential.

        This does not start or scale it. Returned credentials are for callers to use in
        transport configuration and must not appear in diagnostic logs.
        """
        if not settings.exoscale_inference_api_key:
            raise LLMManagementError(
                "EXOSCALE_INFERENCE_API_KEY must be set for managed inference"
            )
        client = self.client()
        deployment = self.find_deployment(client)
        if deployment is None:
            raise DeploymentNotFoundError(self.config.slug)
        deployment_url = deployment.get("deployment-url", "")
        api_key = settings.exoscale_inference_api_key
        return ConnectionInfo(url=deployment_url, api_key=api_key)

    def query_status(self) -> DeploymentQueryResult:
        """
        Read existence, replicas and connection details without starting inference.

        Return an empty absent state when identity is missing. Attach the configured
        inference key, or an empty key when it is not configured, without querying cloud
        credentials. Replica status is provider metadata, not an HTTP health probe.
        """
        client = self.client()
        deployment = self.find_deployment(client)
        if deployment is None:
            return DeploymentQueryResult(
                exists=False, replicas=0, deployment_url="", api_key=""
            )
        return DeploymentQueryResult(
            exists=True,
            replicas=deployment.get("replicas", 0),
            deployment_url=deployment.get("deployment-url", ""),
            api_key=settings.exoscale_inference_api_key,
        )

    def delete_deployment(self) -> None:
        """
        Delete existing deployment identity without deleting its imported model.
        """
        client = self.client()
        deployment = self.find_deployment(client)
        if deployment is None:
            raise DeploymentNotFoundError(self.config.slug)
        rich.print(f"Deleting deployment {self.config.deployment_name}...")
        op = client.delete_deployment(id=deployment["id"])
        client.wait(op["id"])
        rich.print(f"Deployment {self.config.deployment_name} deleted.")

    def create_or_resume(self, *, replicas: int | None = None) -> None:
        """
        If the deployment does not exist, create it.
        If it exists and is scaled to zero, resume it.
        An explicit replica count also resizes an already running deployment.
        """
        client = self.client()
        deployment = self.find_deployment(client)
        if deployment is None:
            self.create_deployment(replicas=replicas)
        elif deployment.get("replicas", 0) == 0 or (
            replicas is not None and deployment.get("replicas", 0) != replicas
        ):
            self.resume_from_zero(replicas=replicas)
        else:
            rich.print(f"Deployment {self.config.deployment_name} is already running.")

    def get_pydantic_ai_model(self) -> OpenAIChatModel:
        """
        Build an OpenAI-compatible adapter from existing connection details.

        Do not ensure readiness or own the adapter's subsequent inference requests.
        """
        info = self.connection_info()
        return OpenAIChatModel(
            self.config.model,
            provider=OpenAIProvider(api_key=info.api_key, base_url=info.url),
        )

    def test_basic_deployment(self) -> bool:
        """
        Test the deployment by asking the LLM for the capital of France.
        """
        model = self.get_pydantic_ai_model()

        agent = Agent(
            model,
            system_prompt="Given the name of a country, return the capital city of that country. e.g. Germany, return 'Berlin'",
        )
        result = agent.run_sync("France")
        output = result.output
        if output.lower() == "paris":
            return True
        raise LLMManagementError(f"Expected city='Paris' but got city='{output}'")

    def test_instruct_deployment(self) -> bool:
        """
        Check structured capital-city output from the existing deployment.

        Return True on the expected answer or raise a management error. This validates
        transport and structured output, not model accuracy across a dataset.
        """
        model = self.get_pydantic_ai_model()

        class CapitalCity(PydanticBaseModel):
            country: str
            city: str

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

    def logs(self, tail: int) -> str:
        """
        Format provider state details and recent logs for an existing deployment.
        """
        client = self.client()
        deployment = self.find_deployment(client)
        if deployment is None:
            raise DeploymentNotFoundError(self.config.slug)
        state = deployment.get("state", "unknown")
        details = deployment.get("state-details", "")
        lines = [f"Deployment {self.config.slug} (state: {state})"]
        if details:
            lines.append(f"Details: {details}")
        lines.append("")
        log_data = client.get_deployment_logs(id=deployment["id"], tail=tail)
        for entry in log_data.get("logs", []):
            lines.append(f"{entry.get('time', '')}  {entry.get('message', '')}")
        return "\n".join(lines)
