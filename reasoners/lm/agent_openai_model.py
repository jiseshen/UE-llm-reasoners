import cv2
import numpy as np
import io
import base64
from PIL import Image
import time
from typing import Optional, Union
import json

from .openai_model import OpenAIModel, GenerateOutput


# PROMPT_TEMPLATE_ANSWER = 'Your response need to be function calling format'
# PROMPT_TEMPLATE_CONTINUE = 'Your response need to be function calling format'

class AgentOpenAIModel(OpenAIModel):
    def __init__(self, model: str, additional_prompt: str = "ANSWER", **kwargs):
        super().__init__(model, **kwargs)
        self.additional_prompt = additional_prompt

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
        system_prompt: Optional[Union[str, list[str]]],
        user_prompt: Optional[Union[str, list[str]]],
        images: Optional[Union[str, list[str], np.ndarray, list[np.ndarray]]] = None,
        functions: Optional[dict] = None,
        max_tokens: int = None,
        top_p: float = 1.0,
        num_return_sequences: int = 1,
        rate_limit_per_min: Optional[int] = 20,
        stop: Optional[str] = None,
        logprobs: Optional[int] = None,
        temperature=None,
        additional_prompt=None,
        retry=64,
        action_history: Optional[list[str]] = None,
        **kwargs,
    ) -> GenerateOutput:

        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        temperature = self.temperature if temperature is None else temperature
        logprobs = 0 if logprobs is None else logprobs

        # if isinstance(prompt, list):
        #     assert len(prompt) == 1  # @zj: why can't we pass a list of prompts?
        #     prompt = prompt[0]
        # if additional_prompt is None and self.additional_prompt is not None:
        #     additional_prompt = self.additional_prompt
        # elif additional_prompt is not None and self.additional_prompt is not None:
        #     print("Warning: additional_prompt set in constructor is overridden.")
        # if additional_prompt == "ANSWER":
        #     prompt = PROMPT_TEMPLATE_ANSWER + prompt
        # elif additional_prompt == "CONTINUE":
        #     prompt = PROMPT_TEMPLATE_CONTINUE + prompt

        messages = [{"role": "system", "content": system_prompt}]

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

        # check if the model supports vision/multimodal inputs
        supports_vision = False
        multimodal_models = ["gpt-4o", "gpt-4o-mini", "o1", "o1-mini"]
        model_name = self.model.lower()
        if model_name in multimodal_models:
            supports_vision = True

        if images and not supports_vision:
            raise ValueError(f"Model {self.model} does not support vision/multimodal inputs")

        for i in range(1, retry + 1):
            try:
                if rate_limit_per_min is not None:
                    time.sleep(60 / rate_limit_per_min)

                if is_instruct_model:
                    user_content = []
                    if action_history:
                        user_content.append({"type": "text", "text": f"Your action history is: {action_history}"})
                    # build the message content
                    if user_prompt:
                        user_content.append({"type": "text", "text": user_prompt})
                    if images and supports_vision:
                        if isinstance(images, str):
                            images = [images]
                        for image in images:
                            # If image is already a base64 string, use it directly
                            if isinstance(image, str):
                                img_data = image
                            else:
                                img_data = self._process_image_to_base64(image)
                            user_content.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}
                            })

                    messages.append({"role": "user", "content": user_content if len(user_content) > 1 else user_content[0]["text"]})

                    with open("messages.json", "w") as f:
                        json.dump(messages, f)

                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        n=num_return_sequences,
                        tools=functions if functions else [],
                        tool_choice="required"
                    )
                    # process the returned results
                    results = []
                    for choice in response.choices:
                        if hasattr(choice, 'function') and choice.function:
                            results.append({
                                "function_call": {
                                    "name": choice.function.name,
                                    "arguments": choice.function.arguments
                                }
                            })
                        else:
                            results.append(choice.message.tool_calls)

                    return results
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        n=num_return_sequences,
                        logprobs=0,
                        **kwargs,
                    )
                    return response.choices[0].message.content

            except Exception as e:
                print(f"An Error Occurred: {e}, sleeping for {i} seconds")
                time.sleep(i)

        raise RuntimeError(
            "GPTCompletionModel failed to generate output, even after 64 tries"
        )

    def generate(
        self,
        system_prompt: Optional[Union[str, list[str]]],
        user_prompt: Optional[Union[str, list[str]]],
        images: Optional[Union[str, list[str], np.ndarray, list[np.ndarray]]] = None,
        max_tokens: int = None,
        top_p: float = 1.0,
        num_return_sequences: int = 1,
        rate_limit_per_min: Optional[int] = 20,
        logprobs: Optional[int] = None,
        temperature=None,
        additional_prompt=None,
        retry=64,
        **kwargs,
    ) -> GenerateOutput:

        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        temperature = self.temperature if temperature is None else temperature
        logprobs = 0 if logprobs is None else logprobs

        supports_vision = False
        multimodal_models = ["gpt-4o", "gpt-4o-mini", "o1", "o1-mini"]
        model_name = self.model.lower()
        if model_name in multimodal_models:
            supports_vision = True

        messages = [{"role": "system", "content": system_prompt}]
        user_content = []
        if user_prompt:
            user_content.append({"type": "text", "text": user_prompt})
        if images and supports_vision:
            if isinstance(images, str):
                images = [images]
            for image in images:
                # If image is already a base64 string, use it directly
                if isinstance(image, str):
                    img_data = image
                else:
                    img_data = self._process_image_to_base64(image)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}
                })
        messages.append({"role": "user", "content": user_content if len(user_content) > 1 else user_content[0]["text"]})

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
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        n=num_return_sequences,
                    )
                    return GenerateOutput(
                        text=[choice.message.content for choice in response.choices],
                        log_prob=None,
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        n=num_return_sequences,
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
