import time
import uuid
from typing import AsyncGenerator
import logging
import re
import collections
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

async def format_messages(messages: list[dict], role_info : features.RoleInfo) -> tuple[str, str]:
    processed = collections.defaultdict(str)
    for i, msg in enumerate(messages):
        contents = []
        if isinstance(msg["content"], list):
            for cont in msg["content"]:
                if isinstance(cont, str):
                    contents.append(cont)
        else:
            contents.append(msg["content"])
        
        for cont in contents:
            if "<|removeRole|>" in cont:
                cont = cont.replace("<|removeRole|>\n", "").replace("<|removeRole|>", "")
            else:
                role : str = msg.get("role", "")
                role = getattr(role_info, role.lower(), role_info.system)
                cont = f"\b{role}:{cont}"
            
            cont = re.sub(r"\n{2,}", r"\n", cont).strip()
            cont = re.sub(r"(^[ \u3000]+|[ \u3000]+$)", "", cont)
            processed[i] += cont + "\n"
        
        processed[i] = processed[i].strip()

    full = False
    output1, output2 = "", ""
    first_length = len(processed[0])
    for prompt in reversed(processed[1:]):
        if not full:
            if len(output2) + len(prompt) < defines.PROMPT_CHARS_LIMIT:
                output2 = prompt + "\n" + output2
            else:
                full = True
        
        if full:
            if first_length + len(output1) + len(prompt) < defines.PROMPT_CHARS_LIMIT:
                output2 = prompt + "\n" + output2
            else:
                break
    
    output1 = processed[1:] + "\n" + output1
    return output1.strip(), output2.strip()

async def send_message(messages: list[dict], api_key: str, model : str, reasoning : bool = False) -> AsyncGenerator[dict, None]:
    client = CLIENTS.get(api_key, fal_client.AsyncClient(api_key)) if api_key else next_client()

    feat = features.process_features(messages)
    prompt1, prompt2 = await format_messages(messages, feat.ROLE)
    request_id = f"chatcmpl-{uuid.uuid4()}"
    error_message = ""

    print(f"System Prompt({len(prompt1)}): \n{prompt1}")
    print(f"User Prompt({len(prompt2)}): \n{prompt2}")
    print("Response:")

    try:
        assert len(prompt1) < defines.PROMPT_CHARS_LIMIT and len(prompt2) < defines.PROMPT_CHARS_LIMIT, f"Prompt too long: {len(prompt1)} + {len(prompt2)}"

        handler = client.stream(
            "fal-ai/any-llm",
            arguments={
                "system_prompt": prompt1,
                "prompt" : prompt2,
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
    except (fal_client.client.FalClientError, AssertionError) as e:
        error_message = str(e)
        print("")
        logger.error(f"Error: {e}", exc_info=True)
    except httpx_sse._exceptions.SSEError as e:
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
