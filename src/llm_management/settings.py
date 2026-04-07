from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_PATH = Path("conf/exoscale.toml")


class ExoscaleCredentials(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EXOSCALE_", env_file=".env")

    api_key: str = ""
    api_secret: str = ""
    huggingface_token: str = ""


settings = ExoscaleCredentials()
