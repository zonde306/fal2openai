import time
import json
import logging
import defines
import fal
import middleware
import blacksheep

logger = logging.getLogger(__name__)
app = blacksheep.Application()

app.middlewares.append(middleware.check_authorization)


@blacksheep.get("/v1/models")
async def models(request : blacksheep.Request) -> blacksheep.Response:
    return blacksheep.json({
        "object": "list",
        "data": [
            {
                "id": model,
                "object": "model",
                "name": model,
                "created": int(time.time()),
                "owned_by": "fal",
            }
            for model in defines.MODELS
        ],
    })

@blacksheep.post("/v1/chat/completions")
async def chat_completions(request : blacksheep.Request) -> blacksheep.Response:
    data : dict = await request.json()
    model : str = data.get("models", None)
    if model not in defines.MODELS:
        logging.error(f"Invalid model: {model} only {defines.MODELS} are allowed")
        return blacksheep.Response(400, body="Invalid model")
    
    messages : list = data.get("messages", [])
    if not messages:
        logging.error(f"Invalid messages: {messages}")
        return blacksheep.Response(400, body="Invalid messages")
    
    reasoning : bool = data.get("reasoning", True)
    api_key : str = data.get("api_key", "")

    if data.get("stream", False):
        async def streaming():
            async for chunk in fal.send_message(messages, api_key, model, reasoning):
                yield f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"
        return blacksheep.Response(200, content=blacksheep.StreamedContent(b"text/plain", streaming))
    
    return blacksheep.json(await fal.send_message_sync(messages, api_key, model, reasoning))
