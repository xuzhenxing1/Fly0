import json
import re
import requests
from typing import Optional, List
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def chat(self, messages: List[dict], stream: bool = False, **kwargs) -> str:
        pass
    
    @abstractmethod
    def check_service(self) -> bool:
        pass


class OllamaProvider(LLMProvider):
    def __init__(self, model: str, url: str = "http://localhost:11434"):
        self.model = model
        self.url = url
    
    def check_service(self) -> bool:
        try:
            response = requests.get(self.url, timeout=5)
            return response.status_code == 200
        except requests.ConnectionError:
            return False
    
    def chat(self, messages: List[dict], stream: bool = False, **kwargs) -> str:
        data = {
            "model": self.model,
            "messages": messages,
            "stream": stream
        }
        
        data.update(kwargs)
        
        response = requests.post(
            f"{self.url}/api/chat",
            json=data,
            stream=stream,
            timeout=120
        )
        
        if stream:
            response_text = ""
            for chunk in response.iter_lines():
                if chunk:
                    try:
                        json_data = json.loads(chunk)
                        if "message" in json_data and "content" in json_data["message"]:
                            content = json_data["message"]["content"]
                            response_text += content
                            print(content, end="", flush=True)
                    except json.JSONDecodeError:
                        continue
            print()
            return response_text
        else:
            result = response.json()
            return result.get("message", {}).get("content", "")
    
    def chat_with_image(self, prompt: str, image_base64: str, stream: bool = False) -> str:
        """Chat with image using Ollama API"""
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "images": [image_base64],
            "stream": stream
        }
        
        response = requests.post(
            f"{self.url}/api/chat",
            json=data,
            stream=stream,
            timeout=120
        )
        
        if stream:
            response_text = ""
            for chunk in response.iter_lines():
                if chunk:
                    try:
                        json_data = json.loads(chunk)
                        if "message" in json_data and "content" in json_data["message"]:
                            content = json_data["message"]["content"]
                            response_text += content
                            print(content, end="", flush=True)
                    except json.JSONDecodeError:
                        continue
            print()
            return response_text
        else:
            result = response.json()
            return result.get("message", {}).get("content", "")


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, base_url: str, model: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
    
    def check_service(self) -> bool:
        return self.client is not None
    
    def chat(self, messages: List[dict], stream: bool = False, **kwargs) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get('temperature', 0),
            stream=stream
        )
        
        if stream:
            response = ""
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    response += content
                    print(content, end="", flush=True)
            print()
            return response
        else:
            return completion.choices[0].message.content
    
    def chat_with_image(self, prompt: str, image_base64: str, stream: bool = False) -> str:
        """Chat with image using OpenAI-compatible API"""
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=20,
            temperature=0,
            stream=stream
        )
        
        if stream:
            response = ""
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    response += content
                    print(content, end="", flush=True)
            print()
            return response
        else:
            return completion.choices[0].message.content


class LLMInterface:
    
    def __init__(self, sys_prompt_path: str, config):
        self.config = config
        self.provider = self._create_provider()
        
        with open(sys_prompt_path, 'r', encoding='utf-8') as f:
            self.base_system_prompt = f.read()
        
        self.chat_history = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": "向上飞10米"},
            {"role": "assistant", "content": 
             """
                ```python
                drone.fly_to([drone.get_position()[0], drone.get_position()[1], drone.get_position()[2] + 10])
                ```
                This code uses the fly_to() function to move the drone 10 units higher than its current position.
                It retrieves the current position using get_position(), then creates a new position with the same X and Y coordinates but the Z coordinate increased by 10.
                The drone then flies to this new position using fly_to().
             """
            }
        ]
    
    def _create_provider(self) -> LLMProvider:
        """Create LLM provider based on configuration"""
        api_type = self.config.get("API_TYPE", "ollama")
        
        if api_type == "ollama":
            control_config = self.config.get_control_config()
            return OllamaProvider(
                self.config.get("OLLAMA_MODEL", "qwen3-vl:4b-instruct"),
                control_config.get("base_url", "http://localhost:11434")
            )
        elif api_type == "openai":
            return OpenAIProvider(
                self.config.get("OPENAI_API_KEY", ""),
                self.config.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                self.config.get("OPENAI_MODEL", "qwen2.5-vl-72b-instruct")
            )
        elif api_type == "vllm":
            return OpenAIProvider(
                self.config.get("OPENAI_API_KEY", "EMPTY"),
                self.config.get("VLLM_BASE_URL", "http://192.168.10.4:8000/v1"),
                self.config.get("VLLM_MODEL", "qwen2.5-vl-72b-instruct")
            )
        else:
            raise ValueError(f"Unsupported API type: {api_type}")
    
    def _get_system_prompt(self, mode: str = "flight") -> str:
       
        return self.base_system_prompt
    
    def ask(self, prompt: str, mode: str = "flight") -> str:
        self.chat_history[0]["content"] = self._get_system_prompt(mode)
        self.chat_history.append({"role": "user", "content": prompt})
        
        if not self.provider.check_service():
            print("LLM service unavailable")
            return ""
        
        response = self.provider.chat(self.chat_history, stream=True)
        
        self.chat_history.append({"role": "assistant", "content": response})
        
        return response
    
    def ask_with_image(self, prompt: str, image_base64: str, stream: bool = False) -> str:
        """Ask LLM with image"""
        if not self.provider.check_service():
            print("LLM service unavailable")
            return ""
        
        if hasattr(self.provider, 'chat_with_image'):
            return self.provider.chat_with_image(prompt, image_base64, stream)
        else:
            print("Provider does not support image input")
            return ""
    
    @staticmethod
    def extract_code(content: str) -> Optional[str]:
        code_block = re.search(r"```python(.*?)```", content, re.DOTALL)
        return code_block.group(1).strip() if code_block else None
    
    def process(self, command: str, mode: str = "flight") -> Optional[str]:
        response = self.ask(command, mode)
        
        if mode == "flight":
            return self.extract_code(response)
        else:
            return response
    
    def clear_history(self):
        self.chat_history = [self.chat_history[0]]

