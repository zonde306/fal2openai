#!/bin/bash

# 你的API KEY，多个用半角逗号分隔
export API_KEYS=

# 连接用的密码
export AUTHORIZATION_TOKEN=

# 端口号
export PORT=25100

python3 -m uvicorn app:app --port ${PORT}
