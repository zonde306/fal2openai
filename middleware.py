import defines
import blacksheep

async def check_authorization(request: blacksheep.Request, next_handler):
    token = request.headers.get_first(b"Authorization") or b""
    if token.startswith(b"Bearer "):
        token = token[7:]
    
    if token.decode("ascii") != defines.AUTHORIZATION_TOKEN:
        raise blacksheep.HTTPException(401, "Unauthorized")
    
    return await next_handler(request)
