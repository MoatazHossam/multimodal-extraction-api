# Multimodal Extraction API

A portable, on-premises FastAPI foundation for converting Arabic or English text into
workflow-specific structured JSON with a locally hosted Ollama model. This milestone accepts text
only; audio transcription, OCR, uploads, persistence, authentication, and external integrations are
deliberately out of scope.

## Architecture

```text
Input (text)
    ↓
Text extraction (future adapters for OCR and transcription)
    ↓
Normalized text
    ↓
Workflow + provider-neutral extraction service
    ↓
Validated structured JSON
```

HTTP controllers know only the extraction service. The service selects a workflow, each workflow
owns its prompt and Pydantic output model, and the `AIProvider` interface isolates Ollama. Adding a
workflow means defining a `Workflow` and registering it in `app/main.py`; it does not require a new
provider or controller.

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

Planned names such as `voice_action`, `meeting_action`, `email_action`, and `task_action` can reuse
the same endpoint and provider contract when their schemas and prompts are defined.
