from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_PATH = Path("conf/exoscale.toml")


class ExoscaleCredentials(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    exoscale_api_key: str = ""
    exoscale_api_secret: str = ""
    huggingface_token: str = ""
    server_role: str = "test"
    auth_token: str = ""


settings = ExoscaleCredentials()
