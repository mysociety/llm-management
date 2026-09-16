from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .foi.model_spec import QUESTION_SLICE_MODEL, QUESTION_SLICE_REVISION

CONFIG_PATH = Path("conf/exoscale.toml")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", hide_input_in_errors=True
    )

    exoscale_api_key: str = ""
    exoscale_api_secret: str = ""
    huggingface_token: str = ""
    server_role: str = "test"
    auth_tokens: dict[str, str] = Field(default_factory=dict)
    question_slice_model: str = QUESTION_SLICE_MODEL
    question_slice_revision: str = QUESTION_SLICE_REVISION
    question_slice_deployment: str = "question_slice_v2"
    question_slice_gpu_enabled: bool = True
    foi_topic_deployment: str = "foi_topic_v2"
    classifier_batch_size: int = Field(default=8, ge=1, le=256)
    classifier_max_units: int = Field(default=256, ge=1, le=4096)
    cpu_inference_threads: int = Field(default=1, ge=1)
    classifier_cache_dir: str | None = None
    cpu_inference_preload: bool = False


settings = Settings()
