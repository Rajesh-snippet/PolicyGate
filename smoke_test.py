from app.config import get_settings
from app.judge.client import build_judge_client

client = build_judge_client(get_settings())
print(client.complete('Reply with JSON only: {"ok": true}', "Say ok"))