#!/usr/bin/env bash
# Быстрые проверки моста. Запущенный `python -m bridge.cli serve` предполагается.
set -euo pipefail

BASE=${BASE:-http://127.0.0.1:8787}

echo "— health —"; curl -s "$BASE/health" | python3 -m json.tool

echo "— простой запрос —"
curl -s "$BASE/v1/chat" -H 'content-type: application/json' \
  -d '{"prompt":"Одним предложением: что такое мост?"}' | python3 -m json.tool

echo "— OpenAI-совместимый запрос —"
curl -s "$BASE/v1/chat/completions" -H 'content-type: application/json' \
  -d '{"model":"deepseek-chat","messages":[{"role":"system","content":"Отвечай кратко."},{"role":"user","content":"Привет"}]}' \
  | python3 -m json.tool

echo "— стриминг (SSE) —"
curl -N -s "$BASE/v1/chat/completions" -H 'content-type: application/json' \
  -d '{"stream":true,"messages":[{"role":"user","content":"Посчитай до пяти"}]}'

echo
echo "— продолжение диалога —"
SESSION=$(curl -s "$BASE/v1/chat" -H 'content-type: application/json' -d '{"prompt":"Запомни число 42"}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["session_id"])')
curl -s "$BASE/v1/chat" -H 'content-type: application/json' \
  -d "{\"prompt\":\"Какое число я просил запомнить?\",\"session_id\":\"$SESSION\"}" | python3 -m json.tool
