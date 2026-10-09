from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from .foi.model_spec import FOIModelSettings, QUESTION_SLICE_DEPLOYMENT

CONFIG_PATH = Path("conf/exoscale.toml")


class Settings(FOIModelSettings):
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", hide_input_in_errors=True
    )

    exoscale_api_key: str = ""
    exoscale_api_secret: str = ""
    compute_state_dir: Path = Path(".state/compute")
    huggingface_token: str = ""
    server_role: str = "test"
    auth_tokens: dict[str, str] = Field(default_factory=dict)
    question_slice_deployment: str = QUESTION_SLICE_DEPLOYMENT
    question_slice_gpu_enabled: bool = True
    foi_topic_deployment: str = "foi_topic_v2"
    classifier_batch_size: int = Field(default=8, ge=1, le=256)
    classifier_max_units: int = Field(default=256, ge=1, le=4096)
    cpu_inference_threads: int = Field(default=1, ge=1)
    classifier_cache_dir: str | None = None
    cpu_idle_timeout_minutes: int = Field(default=15, ge=1)
    presidio_spacy_model: str = "en_core_web_sm"
    presidio_score_threshold: float = Field(default=0.5, ge=0, le=1)


settings = Settings()
