
REM 你的API KEY，多个用半角逗号分隔
set API_KEYS=

REM 连接用的密码
set AUTHORIZATION_TOKEN=

REM 端口号
set PORT=25100

python -m uvicorn app:app --port %PORT%
