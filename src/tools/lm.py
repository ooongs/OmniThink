import random
import requests
import threading
import time
import dspy
import os
from openai import OpenAI
from zhipuai import ZhipuAI
from typing import Optional, Literal, Any
from dashscope import Generation

# This code is originally sourced from Repository STORM
# URL: [https://github.com/stanford-oval/storm]


class OpenAIModel_dashscope(dspy.OpenAI):
    """A wrapper class for dspy.OpenAI."""

    def __init__(
            self,
            model: str = "gpt-4o",
            max_tokens: int = 2000,
            api_key: Optional[str] = None,
            base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
            system_prompt: Optional[str] = None,
            enable_cache: bool = False,
            timeout: int = 1000,
            max_retries: int = 10,
            **kwargs
    ):
        super().__init__(model=model, api_key=api_key, api_base=base_url, **kwargs)
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.system_prompt = system_prompt
        self.enable_cache = enable_cache
        self.timeout = timeout
        self.max_retries = max_retries
        self.request_kwargs = {
            key: value
            for key, value in kwargs.items()
            if value is not None and key not in {"api_base", "api_provider", "api_version"}
        }
        self._token_usage_lock = threading.Lock()
        self.max_tokens = max_tokens
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def log_usage(self, response):
        """Log the total tokens from the OpenAI API response."""
        usage_data = response.get('usage')
        if usage_data:
            with self._token_usage_lock:
                self.prompt_tokens += usage_data.get('prompt_tokens', usage_data.get('input_tokens', 0))
                self.completion_tokens += usage_data.get('completion_tokens', usage_data.get('output_tokens', 0))

    def get_usage_and_reset(self):
        """Get the total tokens used and reset the token usage."""
        usage = {
            self.kwargs.get('model') or self.kwargs.get('engine'):
                {'prompt_tokens': self.prompt_tokens, 'completion_tokens': self.completion_tokens}
        }
        self.prompt_tokens = 0
        self.completion_tokens = 0

        return usage

    def _build_messages(self, prompt: str):
        messages = []
        if self.system_prompt:
            if self.enable_cache:
                system_content = [{
                    "type": "text",
                    "text": self.system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }]
            else:
                system_content = self.system_prompt
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": prompt})
        return messages

    def __call__(
            self,
            prompt: str,
            only_completed: bool = True,
            return_sorted: bool = False,
            **kwargs,
    ) -> list[dict[str, Any]]:
        """Copied from dspy/dsp/modules/gpt3.py with the addition of tracking token usage."""

        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        call_url = f"{self.base_url}/chat/completions"
        lm_key = self.api_key or os.getenv('DASHSCOPE_API_KEY') or os.getenv('LM_KEY')
        if not lm_key:
            raise RuntimeError("Set DASHSCOPE_API_KEY or LM_KEY before calling DashScope models.")
        HEADERS = {
            'Content-Type': 'application/json',
            "Authorization": f"Bearer {lm_key}"
        }

        payload = dict(
            model=self.model,
            messages=self._build_messages(prompt),
            max_tokens=self.max_tokens,
            stream=False,
            **self.request_kwargs,
        )
        last_error = None
        for _ in range(self.max_retries):
            try:
                ret = requests.post(call_url, json=payload,
                                    headers=HEADERS, timeout=self.timeout)
                if ret.status_code != 200:
                    raise Exception(f"http status_code: {ret.status_code}\n{ret.content}")
                ret_json = ret.json()
                for output in ret_json['choices']:
                    if output['finish_reason'] not in ['stop', 'function_call']:
                        raise Exception(f'openai finish with error...\n{ret_json}')
                self.log_usage(ret_json)
                return [ret_json['choices'][0]['message']['content']]
            except Exception as e:
                last_error = e
                print(f"请求失败: {e}. 尝试重新请求...")
                time.sleep(1)
        raise RuntimeError(f"DashScope request failed after {self.max_retries} retries: {last_error}")


class DeepSeekModel(dspy.OpenAI):
    """A wrapper class for dspy.OpenAI."""

    def __init__(
            self,
            model: str = "deepseek-chat",
            api_key: Optional[str] = None,
            **kwargs
    ):
        super().__init__(model=model, api_key=api_key, **kwargs)
        self.model = model
        self.api_key = api_key
        self._token_usage_lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def log_usage(self, response):
        """Log the total tokens from the OpenAI API response."""
        usage_data = response.get('usage')
        if usage_data:
            with self._token_usage_lock:
                self.prompt_tokens += usage_data.get('input_tokens', 0)
                self.completion_tokens += usage_data.get('output_tokens', 0)

    def get_usage_and_reset(self):
        """Get the total tokens used and reset the token usage."""
        usage = {
            self.kwargs.get('model') or self.kwargs.get('engine'):
                {'prompt_tokens': self.prompt_tokens, 'completion_tokens': self.completion_tokens}
        }
        self.prompt_tokens = 0
        self.completion_tokens = 0

        return usage

    def __call__(
            self,
            prompt: str,
            only_completed: bool = True,
            return_sorted: bool = False,
            **kwargs,
    ) -> list[dict[str, Any]]:
        """Copied from dspy/dsp/modules/gpt3.py with the addition of tracking token usage."""

        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        LM_KEY = os.getenv('LM_KEY')
        client = OpenAI(api_key=LM_KEY, base_url="https://api.deepseek.com")

        max_retries = 3
        attempt = 0
        messages = []
        if self.model != "deepseek-reasoner":
            messages.append({"role": "system", "content": "You are a helpful assistant"})
        messages.append({"role": "user", "content": prompt})
        print(messages)
        while attempt < max_retries:
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=False
                )
                choices = response["output"]["choices"]
                break

            except Exception as e:
                delay = random.uniform(0, 3)
                time.sleep(delay)
                attempt += 1

        self.log_usage(response)

        completed_choices = [c for c in choices if c["finish_reason"] != "length"]

        if only_completed and len(completed_choices):
            choices = completed_choices

        completions = [c['message']['content'] for c in choices]

        return completions


class QwenModel(dspy.OpenAI):
    """A wrapper class for dspy.OpenAI."""

    def __init__(
            self,
            model: str = "qwen-max-allinone",
            api_key: Optional[str] = None,
            **kwargs
    ):
        super().__init__(model=model, api_key=api_key, **kwargs)
        self.model = model
        self.api_key = api_key
        self._token_usage_lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def log_usage(self, response):
        """Log the total tokens from the OpenAI API response."""
        usage_data = response.get('usage')
        if usage_data:
            with self._token_usage_lock:
                self.prompt_tokens += usage_data.get('input_tokens', 0)
                self.completion_tokens += usage_data.get('output_tokens', 0)

    def get_usage_and_reset(self):
        """Get the total tokens used and reset the token usage."""
        usage = {
            self.kwargs.get('model') or self.kwargs.get('engine'):
                {'prompt_tokens': self.prompt_tokens, 'completion_tokens': self.completion_tokens}
        }
        self.prompt_tokens = 0
        self.completion_tokens = 0

        return usage

    def __call__(
            self,
            prompt: str,
            only_completed: bool = True,
            return_sorted: bool = False,
            **kwargs,
    ) -> list[dict[str, Any]]:
        """Copied from dspy/dsp/modules/gpt3.py with the addition of tracking token usage."""

        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        messages = [{'role': 'user', 'content': prompt}]
        max_retries = 3
        attempt = 0
        while attempt < max_retries:
            try:
                response = Generation.call(
                    model=self.model, 
                    messages=messages,
                    result_format='message',
                ) 
                choices = response["output"]["choices"]
                break

            except Exception as e:
                delay = random.uniform(0, 10)
                time.sleep(delay)
                attempt += 1

        self.log_usage(response)

        completed_choices = [c for c in choices if c["finish_reason"] != "length"]

        if only_completed and len(completed_choices):
            choices = completed_choices

        completions = [c['message']['content'] for c in choices]

        return completions
