import json
from pathlib import Path
from typing import Any, Dict

constants_path = "../../config/constants.json"


class Config:
    """
    (singleton) lazy load json
    """
    _instance = None
    _config_dict: Dict[str, Any] = None
    _config_path: str = None
    _base_dir = None

    @classmethod
    def _init_config_path(cls, config_path: str = constants_path) -> None:
        """set config file path"""
        if cls._base_dir is None:
            cls._base_dir = Path(__file__).parent

        cls._config_path = (cls._base_dir / config_path).resolve()

    @classmethod
    def _load_config(cls) -> None:
        """load json configurations"""
        if cls._config_dict is not None:
            return

        try:
            # resolve path with pathlib
            config_file = Path(cls._config_path)
            if not config_file.exists():
                raise FileNotFoundError(f"[config] config file not found: \"{cls._config_path}\"")

            # load json
            with open(config_file, 'r', encoding="utf-8") as f:
                cls._config_dict = json.load(f)

            print(f"[config] configuration load successfully form \"{cls._config_path}\".")

        except json.JSONDecodeError as e:
            raise ValueError(f"[config] json resolve failed: {e}")
        except Exception as e:
            raise RuntimeError(f"[config] configuration load failed: {e}")

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """
        get configuration value
        for example: Config.get("database.host") -> "localhost"
        """
        if cls._config_path is None:
            cls._init_config_path(constants_path)

        if cls._config_dict is None:
            cls._load_config()

        keys = key.split('.')
        value = cls._config_dict

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            if default is not None:
                return default
            raise KeyError(f"[config] key not exists: {key}")

