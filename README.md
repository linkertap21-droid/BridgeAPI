# BridgeAPI

Локальный мост: **твой собственный HTTP-апи на ПК, который превращает веб-чат DeepSeek в обычный OpenAI-совместимый эндпоинт.**

Никакого API-ключа и оплаты — вместо запросов к `api.deepseek.com` мост держит управляемый Chromium со страницей `chat.deepseek.com`, вставляет в неё промпт, читает DOM ответа и отдаёт текст наружу по REST (в том числе потоком, SSE). Идея взята из ZeroScript (расширение + Bridge → Roblox Studio), только здесь на конце не Roblox, а DeepSeek, а на входе — любой HTTP-клиент.

```
твой скрипт / curl / SDK-клиент
        │  POST http://127.0.0.1:8787/v1/chat/completions
        ▼
   BridgeAPI (FastAPI, localhost)
        │  очередь генераций (1 браузер = 1 ответ)
        ▼
   Playwright → Chromium с профилем data/browser-profile
        │  ввод в поле чата, чтение innerText ответа
        ▼
   chat.deepseek.com  (залогиненный тобой, без ключа)
```

## Быстрый старт

```bash
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium                              # один раз, ~180 МБ

cp .env.example .env                                      # по желанию, дефолты рабочие

python -m bridge.cli login                                # откроется окно — войди в DeepSeek
python -m bridge.cli serve                                # мост на http://127.0.0.1:8787
python -m bridge.cli chat "напиши хайку про мост"         # проверка тем же клиентом
```

Логин одноразовый: cookies лежат в `data/browser-profile`. Дальше можно `--headless`.

Без браузера вообще (проверка контракта, CI, «упал сайт — надо ли это наш баг?»):

```bash
python -m bridge.cli selftest
BRIDGE_PROVIDER=fake python -m bridge.cli serve     # отвечает заглушкой, полезный статус-код и SSE
```

## API

| Метод | Путь | Назначение |
| --- | --- | --- |
| `POST` | `/v1/chat/completions` | OpenAI-совместимый: `model`, `messages`, `stream`, + `session_id` |
| `POST` | `/v1/chat` | упрощённый: `{"prompt": "...", "session_id": "..."}` |
| `GET` | `/v1/models` | `deepseek-chat`, `deepseek-reasoner` (второй = тумблер DeepThink на странице) |
| `GET` | `/health` | состояние браузера, очереди, статистика, конфиг |
| `POST` | `/v1/auth/login` | поднять headful-окно и дождаться ручного логина (таймаут `BRIDGE_LOGIN_TIMEOUT_S`) |
| `GET` | `/v1/auth/status` | `running` / `logged_in` / текущий URL |
| `POST` | `/v1/auth/logout` | закрыть браузер (забыть аккаунт: удалить `data/browser-profile`) |
| `GET` | `/v1/sessions[/{id}]`, `DELETE /v1/sessions/{id}`, `POST /v1/sessions/clear` | локальная история диалогов |
| `GET` | `/v1/debug/probe` | какие селекторы находятся на странице — первое, что стоит посмотреть, когда всё сломалось |
| `GET` | `/docs` | Swagger UI |

```bash
curl -s localhost:8787/v1/chat -H 'content-type: application/json' \
  -d '{"prompt":"Одним предложением: что такое мост?"}'
# → {"reply":"...","session_id":"srv-92ea...","model":"deepseek-chat","usage":{...,"estimated":true}}

curl -sN localhost:8787/v1/chat/completions -H 'content-type: application/json' \
  -d '{"stream":true,"messages":[{"role":"system","content":"Отвечай кратко."},{"role":"user","content":"Привет"}]}'
# → data: {"choices":[{"delta":{"role":"assistant"}}]} ... data: [DONE]
```

Из Python — любым OpenAI-клиентом, только `base_url`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8787/v1", api_key="не нужен")  # BRIDGE_TOKEN, если задал
r = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "Привет"}],
    extra_body={"session_id": "мой-диалог"},   # опционально: продолжение вместо нового чата
    stream=True,
)
for chunk in r:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

Больше примеров — `examples/curl.sh` и `examples/openai_sdk.py`.

## Как мост ведёт диалог

У веб-чата нет «массива messages», есть поле ввода. Поэтому решает `bridge/engine.py` (`plan()`), и правил три:

1. **страница живая, история не противоречит нашей** → отправляем последнее сообщение дословно, контекст уже на странице;
2. **страница мёртвая** (после рестарта моста или `new chat`) → мост доносит сохранённую историю сам, одним промптом с транскриптом;
3. **клиент прислал другую историю** (редактирование сообщений, как в UI чатов) → доверять странице нельзя, начинается новый диалог с пересборкой контекста.

Ответ на это смотри в поле `rebuilt_context` — оно честно говорит, был ли пересобран контекст. Локальная история живёт в `data/sessions/*.json` (по одному файлу на диалог).

`temperature`/`max_tokens`/`top_p` на странице выключить нечем, поэтому они **игнорируются**, а `usage` — оценка по длине текста (`estimated: true`), а не настоящий tokenizer.

## Конкурентность и ошибки

Одна вкладка = одна генерация. Параллельные запросы становятся в очередь (`BRIDGE_QUEUE_WAIT_S`, `BRIDGE_MAX_WAITERS`); когда очередь переполнена или ожидание истекло — `429 bridge_busy`.

Коды ошибок намеренно разные, потому что реакция на них разная:

| HTTP | `error.type` | что делать |
| --- | --- | --- |
| 503 | `login_required` | `python -m bridge.cli login` (слетели cookies / вышел логин) |
| 503 | `driver_not_ready` | браузер не запущен: `serve` или `login` |
| 502 | `driver_crashed` | вёрстка поменялась или страница упала → `/v1/debug/probe`, править `selectors.json` |
| 504 | `generation_timeout` | ответ не уложился в `BRIDGE_GENERATION_TIMEOUT_S` |
| 429 | `bridge_busy` | мост занят другим запросом |
| 400 | `invalid_request` | некорректное тело запроса |

При потоковой генерации заголовок 200 уже отправлен, поэтому ошибка приходит событием `data: {"error": {...}}` перед `data: [DONE]`.

## Настройка

Всё — переменными окружения или `.env` (полный список с комментариями в `.env.example`). Часто нужное:

- `BRIDGE_TOKEN` — обязателен, если `BRIDGE_HOST` отличается от `127.0.0.1`: иначе любой в сети получит твой аккаунт DeepSeek (`serve` об этом напечатает предупреждение).
- `BRIDGE_HEADLESS=true` — после успешного логина, чтобы не мозолить окно.
- `BRIDGE_POLL_MS` / `BRIDGE_STABLE_POLLS` — как часто и как долго читать DOM; меньше — быстрее реагирует, больше — устойчивее к «задумался».
- `BRIDGE_DATA_DIR` — куда класть профиль и историю.
- `BRIDGE_SELECTORS_FILE` — `selectors.json` (шаблон: `selectors.example.json`).

## Когда DeepSeek обновит вёрстку

Это единственный по-настоящему хрупкий слой проекта, и он изолирован в одном файле. Порядок действий:

```bash
python -m bridge.cli probe          # видно, какие селекторы не находятся, и список кнопок на странице
cp selectors.example.json selectors.json
$EDITOR selectors.json              # подставить новые; каждый ключ можно перечислить через "||"
```

Кодовые пути вёрстки трогать не нужно: драйвер пробует все альтернативы по очереди.

## Структура

```
bridge/
  config.py            настройки из env/.env (свой мини-dotenv, без зависимостей)
  http.py              FastAPI-контракт: эндпоинты, SSE, CORS, токен
  engine.py            очередь, план отправки контекста, дельты, статистика
  sessions.py          JSON-история диалогов + флаг «страница помнит»
  models.py            схемы запросов/ответов (OpenAI-совместимость)
  errors.py            ошибки → HTTP-коды
  drivers/
    base.py            интерфейс ConversationDriver + селекторы + скриптовый драйвер
    deepseek_web.py    Playwright: логин, ввод, чтение DOM, probe
    __init__.py        реестр драйверов
  cli.py               serve / login / status / chat / probe / selftest
```

Слоение принципиальное: `http.py` ничего не знает про браузер, `engine.py` — про DeepSeek, драйвер — про HTTP. Свой бэкенд (например, официальный API по ключу, когда захочется стабильности) = один файл в `drivers/` + одна строка в реестре; `BRIDGE_PROVIDER` переключает.

## Тесты

```bash
pytest                    # 39 тестов, браузер не нужен
```

Покрыто: контракт HTTP (формы ответов, SSE, коды ошибок, 401 при токене, CRUD сессий), планирование контекста (`plan`/`diverges`), очередь и `busy`, разбор селекторов и деградация драйвера без браузера, парсер CLI. Логика DOM намеренно не мокана — её нельзя проверить без живой страницы, для неё есть `probe`.

## Границы честности

- Это **автоматизация веб-интерфейса**, а не официальный API: DeepSeek может запретить такое в правилах сервиса и может сломать вёрсткой в любой день. Для продакшена и нагрузочных сценариев — `api.deepseek.com` с ключом.
- Пока мост генерирует, вкладка занята; не перезагружай и не закрывай её.
- Скорость = скорость чтения человеком: ответ отдаётся снапшотами DOM с частотой `BRIDGE_POLL_MS`, это не нативный streaming.
- Cookies и история лежат у тебя на диске и в `.gitignore`.
