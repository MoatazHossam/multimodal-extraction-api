# Multimodal Extraction API

A portable, on-premises FastAPI foundation for converting Arabic or English text into structured
JSON with a configuration-selected AI provider. Ollama remains the default for fully local,
on-premises operation; Groq is an optional hosted provider for development and testing. The API
also provides independent Arabic/English OCR and Arabic text correction
for written and speech-to-text input. This milestone accepts OCR uploads and text for correction,
action detection, one-call action parsing, or workflow-specific extraction. Audio transcription,
persistence, authentication, action execution, and external integrations are deliberately out of
scope; OCR uploads are processed transiently and never persisted.

## Architecture

```text
Input (text)
    ↓
Independent Arabic correction/normalization (optional)
    or
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
owns its prompt and Pydantic output model, and the `AIProvider` interface isolates provider-specific
HTTP APIs. Action
detection uses the same provider and workflow registry as detailed extraction. Adding a workflow
does not require a new provider.

## API

### Extract text with OCR

`POST /api/v1/ocr/extract` accepts a multipart field named `file` containing a JPEG, PNG, WebP,
or PDF. It uses the local, CPU-only PP-OCRv5 Arabic model (which also recognizes English) and
returns recognized lines in page order:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/ocr/extract \
  -F 'file=@document.pdf;type=application/pdf'
```

```json
{
  "success": true,
  "filename": "document.pdf",
  "page_count": 2,
  "text": "نص الصفحة الأولى\nPage two text",
  "pages": [
    {"page_number": 1, "text": "نص الصفحة الأولى"},
    {"page_number": 2, "text": "Page two text"}
  ]
}
```

This endpoint performs **OCR extraction only**: it does **no grammar correction** and **no action
detection**. It returns OCR output as recognized, without sending it to Qwen, translating it, or
rewriting it. Compose the independent endpoints explicitly when those additional operations are
desired:

```text
/ocr/extract → extracted text → optional /text/correct → optional /actions/parse
```

Uploads are limited to 15 MB and PDFs to 20 pages. PDF pages are rasterized and recognized
sequentially to bound memory and CPU use. Limit violations return HTTP 413; invalid MIME types,
empty files, and malformed files return HTTP 422; OCR engine failures return HTTP 500.

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

### Correct Arabic text

Correct Arabic independently of action detection and parameter extraction. Choose `formal` to
rewrite text as professional Modern Standard Arabic, `asr_repair` to conservatively repair likely
speech-to-text errors without unnecessarily removing dialect, or `asr_formal` to repair an ASR
transcript and then formalize it.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/text/correct \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "وحد اتنينتالتة",
    "mode": "asr_formal"
  }'
```

```json
{
  "success": true,
  "original_text": "وحد اتنينتالتة",
  "corrected_text": "واحد اثنين ثلاثة",
  "changed": true,
  "mode": "asr_formal"
}
```

The service preserves ambiguous tokens rather than guessing and instructs the model never to invent
names, identifiers, contact details, dates, times, amounts, or unsupported facts. Literal numeric
values and email addresses are also checked by the application before a result is returned.
Correction does not invoke the action parser. Provider timeouts return HTTP 504, provider/model or
invalid correction output failures return HTTP 502, and invalid requests return HTTP 422.

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

### Parse actions and parameters

Detect every action and extract the supported actions' parameters in one request. Extraction runs
sequentially in source order, and the endpoint never executes an action. The timestamp must include
a UTC offset and the timezone must be an IANA timezone name.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/actions/parse \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "سو لي اجتماع مع أحمد باچر الساعة عشر وطرش له إيميل بالتفاصيل وذكرني قبل الاجتماع بساعة",
    "reference_datetime": "2026-09-25T05:50:00+04:00",
    "timezone": "Asia/Dubai"
  }'
```

The ordered response combines detection and parameter extraction. `create_task`,
`create_meeting`, `send_email`, and `create_reminder` have `parameter_status: "extracted"` and
their validated parameter object. Other detected types remain in the response with
`parameters: null`, an empty `missing_fields` list, and `parameter_status: "not_supported"` rather
than failing the request. Provider timeouts return HTTP 504, provider/model failures return HTTP
502, and invalid requests return HTTP 422.

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

Copy the example file and select a provider:

```bash
cp .env.example .env
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `AI_PROVIDER` | `ollama` | Provider to use: `ollama` or `groq` |
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Internal Ollama HTTP endpoint |
| `OLLAMA_MODEL` | `qwen3:1.7b` | Installed Ollama model name |
| `OLLAMA_TIMEOUT_SECONDS` | `60` | End-to-end provider request timeout |
| `GROQ_API_KEY` | _(unset)_ | Groq credential; required only when Groq is selected |
| `GROQ_MODEL` | `qwen/qwen3.8-27b` | Groq model name |
| `GROQ_BASE_URL` | `https://api.groq.com/openai/v1` | Groq OpenAI-compatible API endpoint |
| `GROQ_TIMEOUT_SECONDS` | `60` | End-to-end Groq request timeout |
| `PADDLE_TEXT_DETECTION_MODEL` | `PP-OCRv5_mobile_det` | Paddle text detection model |
| `PADDLE_TEXT_RECOGNITION_MODEL` | `arabic_PP-OCRv5_mobile_rec` | Paddle Arabic/English recognition model |

Use `AI_PROVIDER=ollama` for the local/on-prem provider; it requires no credentials and remains
the default. Ollama is accessible only on the shared internal Docker network, and this Compose
project does not publish an Ollama port. Use `AI_PROVIDER=groq` only as an optional hosted
development/testing provider and set `GROQ_API_KEY` in your local `.env`. Never commit that key.
Provider choice is application configuration only: workflows, schemas, corrections, and business
logic use the same `AIProvider` abstraction with either option.

## Local Python development

Python 3.12 and a reachable instance of the selected provider are required for live extraction.

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

The first OCR startup on a host downloads the configured Paddle models and can take longer than
subsequent starts. Compose stores them in the named `paddle_models` volume, so
`docker compose up -d --build` reuses the cache across API image and container rebuilds. Remove
that volume only when you intentionally want Paddle to download the models again.

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
