import os
import uuid
from typing import AsyncGenerator, Optional

import bentoml
from annotated_types import Ge, Le
from typing_extensions import Annotated

MAX_MODEL_LEN = 8192
MAX_TOKENS = 1024
SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

MODEL_ID = "nothingiisreal/MN-12B-Starcannon-v2-fp8-dynamic"


@bentoml.service(
    name="bentovllm-mn-12b-starcannon-v2-fp8-service",
    traffic={
        "timeout": 300,
        "concurrency": 256, # Matches the default max_num_seqs in the VLLM engine
    },
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-l4",
    },
)
class VLLM:

    def __init__(self) -> None:
        from transformers import AutoTokenizer
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        _num_scheduler_steps = os.environ.get("NUM_SCHEDULER_STEPS", "1")
        num_scheduler_steps = int(_num_scheduler_steps)

        ENGINE_ARGS = AsyncEngineArgs(
            model=MODEL_ID,
            max_model_len=MAX_MODEL_LEN,
            num_scheduler_steps=num_scheduler_steps,
            enable_prefix_caching=True
        )

        self.engine = AsyncLLMEngine.from_engine_args(ENGINE_ARGS)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


    @bentoml.api
    async def generate(
        self,
        prompt: str = "Explain superconductors in plain English",
        system_prompt: Optional[str] = SYSTEM_PROMPT,
        max_tokens: Annotated[int, Ge(128), Le(MAX_TOKENS)] = MAX_TOKENS,
    ) -> AsyncGenerator[str, None]:
        from vllm import SamplingParams

        SAMPLING_PARAM = SamplingParams(
            max_tokens=max_tokens,
            ignore_eos=False,
            stop_token_ids=[],
            stop=[],
            min_tokens=max_tokens,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        stream = await self.engine.add_request(uuid.uuid4().hex, prompt, SAMPLING_PARAM)

        cursor = 0
        async for request_output in stream:
            text = request_output.outputs[0].text
            yield text[cursor:]
            cursor = len(text)
