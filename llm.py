import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import os
import requests
from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

class URALLM(LLM):
    llm_url = os.environ.get("URL")
    class Config:
        extra = 'forbid'

    @property
    def _llm_type(self) -> str:
        return "URALLM"

    def _call(
        self,
        inputs: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        payload = {
            "inputs": inputs,
            # "return_full_text":True,
            # "do_sample":True,
            "parameters": {"max_new_tokens":512,
                           "temperature":0.01,
                           "repetition_penalty":1.1,
                           "do_sample":True,
                           "top_k":10
                           },
            "token": os.environ.get("TOKEN")
        }

        headers = {"Content-Type": "application/json"}

        response = requests.post(self.llm_url, json=payload, headers=headers, verify=False)
        response.raise_for_status()

        # print("API Response:", response.json())

        return response.json()['generated_text']  # get the response from the API
        # return response.json().get('generated_text', '')

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"llmUrl": self.llm_url}