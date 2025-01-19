import os
import openai
import numpy as np
from typing import Optional, Union, Literal
import time
import base64
import cv2
import io
from PIL import Image

from reasoners.base import LanguageModel, GenerateOutput
from openai import OpenAI

PROMPT_TEMPLATE_ANSWER = 'Your response need to be ended with "So the answer is"\n\n'
PROMPT_TEMPLATE_CONTINUE = "Please continue to answer the last question, following the format of previous examples. Don't say any other words.\n\n"

class OpenAIModel(LanguageModel):
    def __init__(
        self,
        model: str,
        max_tokens: int = 2048,
        temperature=0.0,
        additional_prompt=None,
        backend: Literal["openai", "sglang"] = "openai",
        is_instruct_model: bool = False,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.backend = backend
        self.additional_prompt = additional_prompt
        self.is_instruct_model = is_instruct_model
        self.__init_client__()

    def __init_client__(self):
        if self.backend == "openai":
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY", None),
            )
        elif self.backend == "sglang":
            self.client = OpenAI(
                base_url=os.getenv("SGLANG_API_URL", None),
            )
        else:
            raise ValueError(f"Invalid backend: {self.backend}")

    def generate(
        self,
        prompt: Optional[Union[str, list[str]]],
        max_tokens: int = None,
        top_p: float = 1.0,
        num_return_sequences: int = 1,
        rate_limit_per_min: Optional[int] = 20,
        stop: Optional[str] = None,
        logprobs: Optional[int] = None,
        temperature=None,
        additional_prompt=None,
        retry=64,
        **kwargs,
    ) -> GenerateOutput:

        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        temperature = self.temperature if temperature is None else temperature
        logprobs = 0 if logprobs is None else logprobs

        if isinstance(prompt, list):
            assert len(prompt) == 1  # @zj: why can't we pass a list of prompts?
            prompt = prompt[0]
        if additional_prompt is None and self.additional_prompt is not None:
            additional_prompt = self.additional_prompt
        elif additional_prompt is not None and self.additional_prompt is not None:
            print("Warning: additional_prompt set in constructor is overridden.")
        if additional_prompt == "ANSWER":
            prompt = PROMPT_TEMPLATE_ANSWER + prompt
        elif additional_prompt == "CONTINUE":
            prompt = PROMPT_TEMPLATE_CONTINUE + prompt

        is_instruct_model = self.is_instruct_model
        if not is_instruct_model:
            # Recheck if the model is an instruct model with model name
            model_name = self.model.lower()
            if (
                ("gpt-3.5" in model_name)
                or ("gpt-4" in model_name)
                or ("instruct" in model_name)
            ):
                is_instruct_model = True

        for i in range(1, retry + 1):
            try:
                # sleep several seconds to avoid rate limit
                if rate_limit_per_min is not None:
                    time.sleep(60 / rate_limit_per_min)
                ### GPT 3.5 and higher use a different API
                if is_instruct_model:
                    messages = [{"role": "user", "content": prompt}]
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        n=num_return_sequences,
                        stop=stop,
                    )
                    return GenerateOutput(
                        text=[choice.message.content for choice in response.choices],
                        log_prob=None,
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        n=num_return_sequences,
                        stop=stop,
                        logprobs=0,
                        **kwargs,
                    )
                    return GenerateOutput(
                        text=[choice["text"] for choice in response.choices],
                        log_prob=[choice["logprobs"] for choice in response["choices"]],
                    )

            except Exception as e:
                print(f"An Error Occured: {e}, sleeping for {i} seconds")
                time.sleep(i)

        # after 64 tries, still no luck
        raise RuntimeError(
            "GPTCompletionModel failed to generate output, even after 64 tries"
        )

    def _process_image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy array image to base64 string.
        
        Args:
            image (np.ndarray): Image array (1 or 3 channels)
            
        Returns:
            str: Base64 encoded image string
        """
        # Convert single channel to 3 channels if needed
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Ensure uint8 type
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
            
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        
        # Convert to base64
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return img_str

    def function_calling(
        self,
        prompt: Optional[Union[str, list[str]]],
        images: Optional[Union[str, list[str], np.ndarray, list[np.ndarray]]] = None,
        functions: Optional[list[dict]] = None,
        function_call: Optional[Union[str, dict]] = None,
        max_tokens: int = None,
        top_p: float = 1.0,
        num_return_sequences: int = 1,
        rate_limit_per_min: Optional[int] = 20,
        stop: Optional[str] = None,
        logprobs: Optional[int] = None,
        temperature=None,
        additional_prompt=None,
        retry=64,
        **kwargs,
    ) -> GenerateOutput:

        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        temperature = self.temperature if temperature is None else temperature
        logprobs = 0 if logprobs is None else logprobs

        if isinstance(prompt, list):
            assert len(prompt) == 1  # @zj: why can't we pass a list of prompts?
            prompt = prompt[0]
        if additional_prompt is None and self.additional_prompt is not None:
            additional_prompt = self.additional_prompt
        elif additional_prompt is not None and self.additional_prompt is not None:
            print("Warning: additional_prompt set in constructor is overridden.")
        if additional_prompt == "ANSWER":
            prompt = PROMPT_TEMPLATE_ANSWER + prompt
        elif additional_prompt == "CONTINUE":
            prompt = PROMPT_TEMPLATE_CONTINUE + prompt

        is_instruct_model = self.is_instruct_model
        if not is_instruct_model:
            # Recheck if the model is an instruct model with model name
            model_name = self.model.lower()
            if (
                ("gpt-3.5" in model_name)
                or ("gpt-4" in model_name)
                or ("instruct" in model_name)
            ):
                is_instruct_model = True

        # 检查多模态支持
        supports_vision = False
        multimodal_models = ["gpt-4o", "gpt-4o-mini", "o1", "o1-mini"]
        model_name = self.model.lower()
        if model_name in multimodal_models:
            supports_vision = True

        if images and not supports_vision:
            raise ValueError(f"Model {self.model} does not support vision/multimodal inputs")

        # Process images if they are numpy arrays
        if isinstance(images, np.ndarray):
            images = [self._process_image_to_base64(images)]
        elif isinstance(images, list) and len(images) > 0 and isinstance(images[0], np.ndarray):
            images = [self._process_image_to_base64(img) for img in images]

        for i in range(1, retry + 1):
            try:
                if rate_limit_per_min is not None:
                    time.sleep(60 / rate_limit_per_min)

                if is_instruct_model:
                    # 构建消息内容
                    content = []
                    if prompt:
                        content.append({"type": "text", "text": prompt})
                    if images and supports_vision:
                        if isinstance(images, str):
                            images = [images]
                        for image in images:
                            # If image is already a base64 string, use it directly
                            if isinstance(image, str):
                                img_data = image
                            else:
                                img_data = self._process_image_to_base64(image)
                            content.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}
                            })

                    messages = [{"role": "user", "content": content if len(content) > 1 else content[0]["text"]}]

                    # 创建API调用参数
                    api_params = {
                        "model": self.model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "n": num_return_sequences,
                        "stop": stop,
                    }

                    # 添加函数调用相关参数
                    if functions:
                        api_params["functions"] = functions
                    if function_call:
                        api_params["function_call"] = function_call

                    response = self.client.chat.completions.create(**api_params)

                    # 处理返回结果
                    results = []
                    for choice in response.choices:
                        if hasattr(choice, 'function_call') and choice.function_call:
                            results.append({
                                "function_call": {
                                    "name": choice.function_call.name,
                                    "arguments": choice.function_call.arguments
                                }
                            })
                        else:
                            results.append(choice.message.content)

                    return GenerateOutput(
                        text=results,
                        log_prob=None,
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        n=num_return_sequences,
                        stop=stop,
                        logprobs=0,
                        **kwargs,
                    )
                    return GenerateOutput(
                        text=[choice["text"] for choice in response.choices],
                        log_prob=[choice["logprobs"] for choice in response["choices"]],
                    )

            except Exception as e:
                print(f"An Error Occurred: {e}, sleeping for {i} seconds")
                time.sleep(i)

        raise RuntimeError(
            "GPTCompletionModel failed to generate output, even after 64 tries"
        )

    def get_next_token_logits(
        self,
        prompt: Union[str, list[str]],
        candidates: Union[list[str], list[list[str]]],
        **kwargs,
    ) -> list[np.ndarray]:
        raise NotImplementedError(
            "GPTCompletionModel does not support get_next_token_logits"
        )

    def get_loglikelihood(
        self, prompt: Union[str, list[str]], **kwargs
    ) -> list[np.ndarray]:
        raise NotImplementedError("GPTCompletionModel does not support get_log_prob")


if __name__ == "__main__":
    model = OpenAIModel(model="gpt-3.5-turbo")
    print("-------OpenAI client-------")
    print(model.generate(["How to go to Shanghai from Beijing?"]))
    print("-------SGLang client-------")
    model = OpenAIModel(
        model="meta-llama/Llama-3.1-8B-Instruct",
        backend="sglang",
        is_instruct_model=True,
    )
    print(model.generate(["How to go to Shanghai from Beijing?"]))
