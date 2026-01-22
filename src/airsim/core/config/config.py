"""Configuration management module"""
import json
import os
from typing import Dict, Any


class Config:
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_vision_config(self) -> Dict[str, Any]:
        api_type = self.get("API_TYPE", "openai")
        
        if api_type == "openai":
            return {
                "enabled": self.get("vision.enabled", False),
                "api_key": self.get("OPENAI_API_KEY", ""),
                "base_url": self.get("OPENAI_BASE_URL", ""),
                "model": self.get("OPENAI_MODEL", "qwen2.5-vl-72b-instruct")
            }
        elif api_type == "ollama":
            return {
                "enabled": self.get("vision.enabled", False),
                "api_key": "",
                "base_url": self.get("vision.base_url", "http://127.0.0.1:11434"),
                "model": self.get("OLLAMA_MODEL", "qwen3-vl:4b-instruct")
            }
        elif api_type == "vllm":
            return {
                "enabled": self.get("vision.enabled", False),
                "api_key": "EMPTY",
                "base_url": self.get("VLLM_BASE_URL", "http://192.168.10.4:8000/v1"),
                "model": self.get("VLLM_MODEL", "qwen2.5-vl-72b-instruct")
            }
        else:
            return {
                "enabled": False,
                "api_key": "",
                "base_url": "",
                "model": ""
            }
    
    def get_control_config(self) -> Dict[str, Any]:
        api_type = self.get("API_TYPE", "openai")
        
        if api_type == "openai":
            return {
                "enabled": self.get("control.enabled", True),
                "api_key": self.get("OPENAI_API_KEY", ""),
                "base_url": self.get("OPENAI_BASE_URL", ""),
                "model": self.get("OPENAI_MODEL", "qwen2.5-vl-72b-instruct")
            }
        elif api_type == "ollama":
            return {
                "enabled": self.get("control.enabled", True),
                "api_key": "",
                "base_url": self.get("control.base_url", "http://127.0.0.1:11434"),
                "model": self.get("OLLAMA_MODEL", "qwen3-vl:4b-instruct")
            }
        elif api_type == "vllm":
            return {
                "enabled": self.get("control.enabled", True),
                "api_key": "EMPTY",
                "base_url": self.get("VLLM_BASE_URL", "http://192.168.10.4:8000/v1"),
                "model": self.get("VLLM_MODEL", "qwen2.5-vl-72b-instruct")
            }
        else:
            return {
                "enabled": True,
                "api_key": "",
                "base_url": "",
                "model": ""
            }
    
    def get_planner_config(self) -> Dict[str, Any]:
        return {
            "lidar_sensors": self.get("planner.lidar_sensors", ["LidarSensor1"])
        }
