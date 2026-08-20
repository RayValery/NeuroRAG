from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """All app configuration, loaded from environment / .env and type-validated."""

    # --- Entrez (PubMed ingestion) ---
    entrez_email: str
    entrez_api_key: str | None = None

    # --- Database ---
    database_url: str       # REQUIRED: postgresql://neurorag:neurorag@localhost:5432/neurorag

    # --- Embedding model ---
    embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO"  # sensible default (768-dim)

    # --- LLM (answer step)
    llm_provider: str | None = None
    llm_api_key: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"      # ignore unknown keys in .env instead of erroring
    )

# One shared instance the rest of the app imports: " from neurorag.config import settings "
settings = Settings()