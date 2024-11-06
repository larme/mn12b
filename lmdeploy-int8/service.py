import uuid
from typing import AsyncGenerator, Optional

import bentoml
from annotated_types import Ge, Le
from typing_extensions import Annotated


MAX_TOKENS = 8192
SYSTEM_PROMPT = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
"""

MODEL_ID = "AuriAetherwiing/MN-12B-Starcannon-v2"


@bentoml.service(
    name="bentovllm-mn-12b-starcannon-v2-int8-service",
    traffic={
        "timeout": 300,
    },
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-l4",
    },
)
class LMDeploy:

    def __init__(self) -> None:
        from transformers import AutoTokenizer
        from lmdeploy.serve.async_engine import AsyncEngine
        from lmdeploy.messages import TurbomindEngineConfig

        engine_config = TurbomindEngineConfig(
            model_format="hf",
            cache_max_entry_count=0.9,
            enable_prefix_caching=True,
            session_len=MAX_TOKENS,
            quant_policy=8,
        )
        self.engine = AsyncEngine(
            MODEL_ID, backend_config=engine_config
        )
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


    @bentoml.api
    async def generate(
        self,
        ctx: bentoml.Context,
        prompt: str = "Explain superconductors in plain English",
        system_prompt: Optional[str] = SYSTEM_PROMPT,
        max_tokens: Annotated[int, Ge(128), Le(MAX_TOKENS)] = MAX_TOKENS,
    ) -> AsyncGenerator[str, None]:

        from lmdeploy import GenerationConfig

        gen_config = GenerationConfig(
            max_new_tokens=max_tokens,
            min_new_tokens=max_tokens,
            ignore_eos=True,
            stop_token_ids=[],
            stop_words=[],
        )

        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)


        session_id = abs(uuid.uuid4().int >> 96)
        stream = self.engine.generate(
            prompt, session_id=session_id, gen_config=gen_config
        )

        async for request_output in stream:
            if await ctx.request.is_disconnected():
                await self.engine.stop_session(session_id)
                await self.engine.end_session(session_id)
                return
            yield request_output.response
