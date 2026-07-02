from __future__ import annotations

import os
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root (scripts/app/core/ -> up 3); .env lives there.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=os.path.join(_ROOT, ".env"), extra="ignore")

    app_name: str = "dynamic-kg"
    environment: str = "dev"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    database_url: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/phd"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4jpassword"

    llm_provider: str = "gemini"

    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    gemini_temps: str = "0.2,0.6,0.9"
    gemini_mock: bool = False

    mistral_api_key: str | None = None
    mistral_model: str = "mistral-small-latest"

    csv_sample_rows: int = 20
    merge_policy_version: str = "v1"

    @property
    def temperature_list(self) -> List[float]:
        temps = []
        for part in self.gemini_temps.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                temps.append(float(part))
            except ValueError:
                continue
        return temps or [0.2, 0.6, 0.9]


settings = Settings()
