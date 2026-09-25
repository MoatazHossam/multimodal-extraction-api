# Multimodal Extraction API

A portable, on-premises FastAPI foundation for converting Arabic or English text into structured
JSON with a locally hosted Ollama model. This milestone accepts text for action detection or
workflow-specific extraction. Audio transcription, OCR, uploads, persistence, authentication,
action execution, and external integrations are deliberately out of scope.

## Architecture

```text
Input (text)
    ↓
Text extraction (future adapters for OCR and transcription)
    ↓
Normalized text
    ↓
Action detector → ordered actions[] → action-specific parameter schemas
    or
Workflow-specific extraction
    ↓
Provider-neutral service + validated structured JSON
```

HTTP controllers know only their application services. Services select a workflow, each workflow
owns its prompt and Pydantic output model, and the `AIProvider` interface isolates Ollama. Action
detection uses the same provider and workflow registry as detailed extraction. Adding a workflow
does not require a new provider.

## API

### Health

```bash
curl http://127.0.0.1:8000/health
```

### Extract text

```bash
curl -X POST http://127.0.0.1:8000/api/v1/extract/text \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "الملف رقم 5678 يحتاج مساعدة في دفع المصروفات الدراسية والأمر عاجل",
    "workflow": "assistance_request"
  }'
```

The response contains the original text, selected workflow, validated workflow data, and the names
of fields for which the source contained no value. Interactive OpenAPI documentation is available
at `http://127.0.0.1:8000/docs`.

### Detect actions

Detect every supported action in Arabic or English text without executing it or extracting its
detailed parameters:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/actions/detect \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Schedule a meeting with Ahmed tomorrow at 10, email him the details, and remind me one hour before."
  }'
```

```json
{
  "success": true,
  "text": "Schedule a meeting with Ahmed tomorrow at 10, email him the details, and remind me one hour before.",
  "actions": [
    {
      "action_type": "create_meeting",
      "source_text": "Schedule a meeting with Ahmed tomorrow at 10"
    },
    {
      "action_type": "send_email",
      "source_text": "email him the details"
    },
    {
      "action_type": "create_reminder",
      "source_text": "remind me one hour before"
    }
  ]
}
```

Supported action types are `create_task`, `create_meeting`, `send_email`, `create_request`,
`create_reminder`, `create_note`, `follow_up`, and `unknown`. Results preserve source order. A
non-actionable input produces one `unknown` item containing the original text.

### Extract action parameters

Extract validated parameters for one previously detected task, meeting, email, or reminder. The
full original text is supplied separately so references such as Arabic `له` or English `him` can
be resolved without expanding the detected action clause. `reference_datetime` (including its UTC
offset) is the sole reference for relative dates, and `timezone` must be an IANA timezone name.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/actions/extract-parameters \
  -H 'Content-Type: application/json' \
  -d '{
    "original_text": "اعمل اجتماع مع أحمد بكرة الساعة 10 وابعتله إيميل بالتفاصيل",
    "action": {
      "action_type": "create_meeting",
      "source_text": "اعمل اجتماع مع أحمد بكرة الساعة 10"
    },
    "reference_datetime": "2026-09-25T04:45:00+04:00",
    "timezone": "Asia/Dubai"
  }'
```

```json
{
  "success": true,
  "action_type": "create_meeting",
  "source_text": "اعمل اجتماع مع أحمد بكرة الساعة 10",
  "parameters": {
    "title": null,
    "attendees": ["أحمد"],
    "date": "2026-09-26",
    "time": "10:00",
    "duration_minutes": null,
    "location": null,
    "agenda": null
  },
  "missing_fields": []
}
```

The endpoint does not execute actions. Values are validated against the selected action schema,
and `missing_fields` reports only execution-critical information. Unsupported action types return
HTTP 422; invalid model output returns HTTP 502.

## Configuration

Copy the example file and adjust it only when your Ollama service or model differs:

```bash
cp .env.example .env
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Internal Ollama HTTP endpoint |
| `OLLAMA_MODEL` | `qwen3:1.7b` | Installed Ollama model name |
| `OLLAMA_TIMEOUT_SECONDS` | `60` | End-to-end provider request timeout |

No credentials are required or committed. Ollama remains accessible only on the shared internal
Docker network; this Compose project does not publish an Ollama port.

## Local Python development

Python 3.12 and a reachable Ollama instance are required for live extraction.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
uvicorn app.main:app --reload
```

Run checks without a live Ollama server (the tests use an in-memory HTTP transport):

```bash
pytest
ruff check .
```

## Docker deployment (VPS or on premises)

Ollama must already be running on the external network under the hostname configured by
`OLLAMA_BASE_URL`. Create the network once if it does not already exist:

```bash
docker network create ai_internal
```

Ensure the separately managed Ollama container is connected to it and that the model is installed,
then deploy the API:

```bash
cp .env.example .env
docker compose build
docker compose up -d
curl http://127.0.0.1:8000/health
```

The API is published at `127.0.0.1:8000`, not on the server's public interfaces. Moving between a
temporary VPS and an on-premises host requires only Docker, the external `ai_internal` network, an
Ollama container with the configured model, and this Compose project. A reverse proxy and HTTPS can
be added in a later milestone without changing the application.

## Adding a workflow

1. Create a Pydantic output model and a `Workflow` implementation under `app/workflows/`.
2. Give it a unique `name` and a prompt that instructs the model not to invent absent data.
3. Register the workflow in the `WorkflowRegistry` in `app/main.py`.
4. Add service and API tests covering valid, missing, and malformed fields.

Additional action-specific extraction workflows can consume the action detector's ordered output
while reusing the same provider contract.
