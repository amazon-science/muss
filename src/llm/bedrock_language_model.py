import boto3
import json
import logging
import time

from botocore.config import Config
from typing import Dict, Optional, Tuple

LLM_ID_CLAUDE_HAIKU = "us.anthropic.claude-3-5-haiku-20241022-v1:0"

MAX_NUM_ATTEMPTS = 5

DEFAULT_MODEL_INFERENCE_ARGS = {
    LLM_ID_CLAUDE_HAIKU: {
        "anthropic_version": "bedrock-2023-05-31",
        "temperature": 0.0,
        "max_tokens": 1000,
        "top_k": 1,
        "stop_sequences": ["\n\nHuman:"],
    }
}

logger = logging.getLogger(__name__)


class BedrockLanguageModel:
    def __init__(self, model_name: str, inference_args: Optional[Dict] = None) -> None:
        self.model_name = model_name
        if inference_args is None:
            inference_args = DEFAULT_MODEL_INFERENCE_ARGS.get(model_name, {})
        self.inference_args = inference_args
        self.client = self._get_bedrock_client()

    @staticmethod
    def _get_bedrock_client():
        session = boto3.Session()
        target_region = "us-west-2"
        retry_config = Config(
            region_name=target_region,
            connect_timeout=10,
            retries={"max_attempts": 1, "mode": "standard"},
        )
        bedrock_client = session.client(
            service_name="bedrock-runtime",
            config=retry_config,
            region_name=target_region,
        )
        return bedrock_client

    def generate_text(
        self, user_prompt: str, sys_prompt: str="", assistant_prompt: Optional[str]=None
    ) -> Tuple[str, str]:
        body_dict = {}
        if self.model_name == LLM_ID_CLAUDE_HAIKU:
            body_dict = {
                "system": sys_prompt,
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
                ]
            }
            if assistant_prompt is not None:
                body_dict["messages"].append(
                    {"role": "assistant", "content": [{"type": "text", "text": assistant_prompt}]}
                )
        body_dict.update(self.inference_args)
        body = json.dumps(body_dict)
        response_text = ""
        error_message = ""
        for i in range(MAX_NUM_ATTEMPTS):
            try:
                response = self.client.invoke_model(
                    body=body, modelId=self.model_name, contentType="application/json", accept="*/*"
                )
                response_body = json.loads(response.get("body").read())
                response_text = response_body["content"][0]["text"]
                error_message = ""
                break
            except Exception as e:
                logger.info(f"exception: {type(e)}")
                error_message = str(e)
                logger.info(error_message)
                if "ThrottlingException" not in str(type(e)):
                    break
                logger.info(f"will try {MAX_NUM_ATTEMPTS - i - 1} more times")
                time.sleep(3**(i + 1))

        return response_text, error_message
