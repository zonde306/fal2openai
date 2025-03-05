import re
import dataclasses

FEATURES_DEFINE = {}

@dataclasses.dataclass
class RoleInfo:
    user: str = "Human"
    assistant: str = "Assistant"
    system: str = "System"
    developer: str = "System"

@dataclasses.dataclass
class Features:
    ROLE : RoleInfo
    SYSTEM_PROMPT: str = ""

def extract_role_info(content : str) -> tuple[RoleInfo, str]:
    if not isinstance(content, str):
        return (RoleInfo(), content)

    pattern = re.compile(r"<roleInfo>([\s\S]*)</roleInfo>", re.MULTILINE)
    
    if matched := pattern.search(content):
        roles = {}
        for line in matched.group(1).split("\n"):
            line = line.strip()
            if not line:
                continue
            key, value = line.split(":", 1)
            roles[key.strip().lower()] = value.strip()
        
        return (
            RoleInfo(**roles),
            re.sub(pattern, "", content)
        )
    
    return (RoleInfo(), content)

def process_features(messages : list) -> Features:
    feats = {}

    for k, v in FEATURES_DEFINE.items():
        on = k in messages[0]["content"]
        feats[v] = on
        if on:
            messages[0]["content"] = messages[0]["content"].replace(f"{k}\n", "").replace(k, "")

    role, cont = extract_role_info(messages[0]["content"])

    system = ""
    if messages[0]["role"] == "system":
        system = cont
        messages.pop(0)
    else:
        messages[0]["content"] = cont

    return Features(role, system, **feats)
