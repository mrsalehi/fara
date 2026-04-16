import argparse
import base64
import io
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image
from transformers import AutoProcessor

from fara.modeling import Qwen2_5_VLForConditionalGeneration


class ChatCompletionsRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Dict[str, Any]]
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False


class ModelServer:
    def __init__(
        self,
        model_path: str,
        model_name: str,
        dtype: str = "auto",
        device: str = "cuda",
        device_map: str = "auto",
        max_new_tokens_default: int = 512,
    ):
        self.model_path = model_path
        self.model_name = model_name
        self.max_new_tokens_default = max_new_tokens_default

        self.torch_dtype = self._resolve_dtype(dtype)
        self.device = device
        self.device_map = device_map

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    @staticmethod
    def _resolve_dtype(dtype: str):
        if dtype == "auto":
            return "auto"
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if dtype not in mapping:
            raise ValueError(f"Unsupported dtype: {dtype}")
        return mapping[dtype]

    @staticmethod
    def _load_image_from_url(url: str) -> Image.Image:
        if url.startswith("data:image"):
            try:
                b64_data = url.split(",", 1)[1]
                raw = base64.b64decode(b64_data)
                return Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception as exc:
                raise ValueError(f"Failed to decode data URL image: {exc}") from exc

        if url.startswith("http://") or url.startswith("https://"):
            try:
                resp = requests.get(url, timeout=20)
                resp.raise_for_status()
                return Image.open(io.BytesIO(resp.content)).convert("RGB")
            except Exception as exc:
                raise ValueError(f"Failed to download image URL: {exc}") from exc

        if url.startswith("file://"):
            path = url[len("file://") :]
        else:
            path = url

        try:
            return Image.open(path).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Failed to read local image '{path}': {exc}") from exc

    def _convert_openai_messages(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
        qwen_messages: List[Dict[str, Any]] = []
        images: List[Image.Image] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            out_content: List[Dict[str, Any]] = []

            if isinstance(content, str):
                out_content.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue

                    part_type = part.get("type")
                    if part_type == "text":
                        out_content.append({"type": "text", "text": part.get("text", "")})
                    elif part_type == "image_url":
                        image_url = part.get("image_url", {}).get("url")
                        if not image_url:
                            continue
                        image = self._load_image_from_url(image_url)
                        images.append(image)
                        out_content.append({"type": "image", "image": image})
            else:
                out_content.append({"type": "text", "text": str(content)})

            qwen_messages.append({"role": role, "content": out_content})

        return qwen_messages, images

    def generate(self, req: ChatCompletionsRequest) -> Dict[str, Any]:
        if req.stream:
            raise HTTPException(status_code=400, detail="stream=true is not supported by this server")

        qwen_messages, images = self._convert_openai_messages(req.messages)
        prompt_text = self.processor.apply_chat_template(
            qwen_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        processor_kwargs: Dict[str, Any] = {
            "text": [prompt_text],
            "return_tensors": "pt",
            "padding": True,
        }
        if images:
            processor_kwargs["images"] = images

        model_inputs = self.processor(**processor_kwargs)

        if self.device_map == "auto":
            device = next(self.model.parameters()).device
            model_inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in model_inputs.items()}
        elif self.device.startswith("cuda") or self.device == "cpu":
            model_inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in model_inputs.items()}

        max_new_tokens = (
            req.max_completion_tokens
            if req.max_completion_tokens is not None
            else req.max_tokens
            if req.max_tokens is not None
            else self.max_new_tokens_default
        )

        do_sample = (req.temperature is not None) and (req.temperature > 0)
        generate_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            generate_kwargs["temperature"] = req.temperature
            generate_kwargs["top_p"] = req.top_p if req.top_p is not None else 1.0

        with torch.inference_mode():
            generated = self.model.generate(**model_inputs, **generate_kwargs)

        prompt_len = model_inputs["input_ids"].shape[1]
        generated_ids = generated[:, prompt_len:]
        output_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        prompt_tokens = int(prompt_len)
        completion_tokens = int(generated_ids.shape[1])
        total_tokens = prompt_tokens + completion_tokens

        created = int(time.time())
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": req.model or self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": output_text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }


def build_app(server: ModelServer) -> FastAPI:
    app = FastAPI(title="FARA HF OpenAI-Compatible Server")

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/model")
    async def model_info() -> Dict[str, str]:
        return {"model": server.model_name, "model_url": server.model_path}

    @app.get("/v1/models")
    async def list_models() -> Dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": server.model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionsRequest) -> Dict[str, Any]:
        try:
            return server.generate(req)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve FARA HF model with OpenAI-compatible API")
    parser.add_argument("--model-path", type=str, required=True, help="Path to local model weights")
    parser.add_argument("--model-name", type=str, default="fara-hf-local", help="Model name exposed via API")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "float16", "bfloat16", "float32"],
        default="bfloat16",
        help="Torch dtype for model loading",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Target device (only used when --device-map is not auto)",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Transformers device_map value, e.g., auto or cuda:0",
    )
    parser.add_argument(
        "--max-new-tokens-default",
        type=int,
        default=512,
        help="Default max_new_tokens when not provided by request",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ModelServer(
        model_path=args.model_path,
        model_name=args.model_name,
        dtype=args.dtype,
        device=args.device,
        device_map=args.device_map,
        max_new_tokens_default=args.max_new_tokens_default,
    )
    app = build_app(server)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
