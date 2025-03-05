import time
import uuid
from typing import AsyncGenerator
import logging
import features
import defines
import fal_client
import fal_client.client
import httpx_sse

logger = logging.getLogger(__name__)

CLIENTS : dict[str, fal_client.AsyncClient] = {}
CLIENTS_POLL : int = -1

def setup():
    for key in defines.API_KEYS:
        if k := key.strip():
            CLIENTS[k] = fal_client.AsyncClient(k)
    
    if not CLIENTS:
        logger.info("No API keys available")
    else:
        logger.info(f"Using API keys: {len(CLIENTS)}")

def next_client() -> fal_client.AsyncClient:
    if not CLIENTS:
        logger.error("No API keys available")
        return fal_client.async_client

    global CLIENTS_POLL
    CLIENTS_POLL += 1
    clients = list(CLIENTS.values())
    return clients[CLIENTS_POLL % len(clients)]

async def process_upload(src : str) -> str:
    # Unsupported
    logger.warning(f"Unsupported upload: {src}")
    return ""

async def format_messages(messages: list[dict], role_info : features.RoleInfo) -> tuple[str, list[str]]:
    prompt = ""
    attachments = []
    for msg in messages:
        contents = []
        if isinstance(msg["content"], list):
            for cont in msg["content"]:
                if image_url := cont.get("image_url", {}):
                    if img := image_url.get("url"):
                        attachments.append(await process_upload(img))
                elif isinstance(cont, str):
                    contents.append(cont)
        else:
            contents.append(msg["content"])
        
        for cont in contents:
            if "<|removeRole|>" in cont:
                cont = cont.replace("<|removeRole|>\n", "").replace("<|removeRole|>", "")
                prompt += f"{cont}\n\n"
            else:
                role : str = msg.get("role", "")
                role = getattr(role_info, role.lower(), role_info.system)
                prompt += f"\b{role}: {cont}\n\n"

    return prompt, attachments

async def send_message(messages: list[dict], api_key: str, model : str, reasoning : bool = False) -> AsyncGenerator[dict, None]:
    client = CLIENTS.get(api_key, fal_client.AsyncClient(api_key)) if api_key else next_client()

    feat = features.process_features(messages)
    prompt, _ = await format_messages(messages, feat.ROLE)
    request_id = f"chatcmpl-{uuid.uuid4()}"
    error_message = ""

    print(f"System Prompt: \n{feat.SYSTEM_PROMPT}")
    print(f"User Prompt: \n{prompt}")
    print("Response:")

    try:
        handler = client.stream(
            "fal-ai/any-llm",
            arguments={
                "prompt" : prompt,
                "system_prompt": feat.SYSTEM_PROMPT,
                "reasoning": reasoning,
                "model": model,
            },
        )

        is_reasoning = False
        prev_reasoning = 0
        prev_output = 0
        async for event in handler:
            if event["error"]:
                error_message = event["error"]
                break
            
            # 为什么输出流里会包含上次的输出？
            if event["reasoning"] and len(event["reasoning"]) > prev_reasoning:
                event["reasoning"] = event["reasoning"][prev_reasoning:]
                prev_reasoning += len(event["reasoning"])
            if event["output"] and len(event["output"]) > prev_output:
                event["output"] = event["output"][prev_output:]
                prev_output += len(event["output"])
            
            if reasoning and not is_reasoning and event["reasoning"]:
                is_reasoning = True
                content = f"<thinking>\n{event['reasoning']}"
            elif reasoning and is_reasoning and not event["reasoning"]:
                is_reasoning = False
                content = f"</thinking>\n{event['output']}"
            elif event["reasoning"]:
                content = event["reasoning"]
            else:
                content = event["output"]
            
            yield {
                "id" : request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": content,
                    },
                }],
            }
            print(content, end="")
    except fal_client.client.FalClientError as e:
        error_message = str(e)
        print("")
        logger.error(f"Error: {e}", exc_info=True)
    except httpx_sse._exceptions.SSEError as e:
        print(f"*** SSL Error: {e}")
        # 他们库的问题
        print("")

    if error_message:
        print(f"ERROR: {error_message}")
        yield {
            "id" : request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {
                    "content": f"ERROR: {error_message}",
                },
                "finish_reason": "error",
            }],
        }
    else:
        yield {
            "id" : request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {
                    "content": "",
                },
                "finish_reason": "stop",
            }],
        }

async def send_message_sync(messages: list[dict], api_key: str, model : str, reasoning : bool = False) -> dict:
    content = ""
    error_message = ""
    async for message in send_message(messages, api_key, model, reasoning):
        content += message["choices"][0]["delta"]["content"]
        if message["choices"][0]["finish_reason"] == "error":
            error_message = message["choices"][0]["delta"]["content"]
    
    if error_message:
        return {
            "id" : message["id"],
            "object": "chat.completion",
            "created": message["created"],
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": error_message,
                },
                "finish_reason": "error",
            }],
        }
    
    return {
        "id" : message["id"],
        "object": "chat.completion",
        "created": message["created"],
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content,
            },
            "finish_reason": "stop",
        }],
        "usage": None,
    }

setup()
