from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    app_name: str = "AI Movie Studio"
    app_env: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    studio_data_dir: Path = Path("./data")
    @property
    def projects_dir(self) -> Path:
        return self.studio_data_dir / "projects"

settings = Settings()
