# notebook

An agent harness over your own sources: upload documents, an **analyzer** model
turns each one into an indexable markdown-latex rendition, and a **researcher**
model answers questions over the index with citations back to the original.

See [AGENTS.md](AGENTS.md) for the architecture and roadmap.

## Run

```bash
cargo run
```

Then open http://127.0.0.1:8080, click the model chip in the prompt bar and
choose **Configure**, then add a provider: a base URL, an API style (Anthropic's
`/v1/messages` or anything OpenAI-compatible) and your API key. The server reads
that endpoint's own model list, and you hide the models you don't want and pin
the ones you do. Keys are stored per user, server-side, and never returned to
the browser.

A local endpoint needs no key at all: `http://127.0.0.1:1234/v1` for LM Studio,
`http://localhost:11434/v1` for Ollama.

`.env` (see `.env.example`) is optional and covers server settings only: bind
address, database and upload paths. API keys do not go in it. The one exception
is seeding a headless install, documented in AGENTS.md section 8.

## API

| method | path | |
|---|---|---|
| GET | `/api/health` | roles, providers, key presence, no secrets |
| GET | `/api/me` | the current user (one local user until auth exists) |
| POST | `/api/sources` | multipart, one or many `file` parts, optional `title` and `folder_id` |
| GET | `/api/sources` | library listing |
| GET | `/api/sources/{id}` | one source |
| PATCH | `/api/sources/{id}` | rename, or move between folders |
| GET | `/api/sources/{id}/raw` | the original bytes, served inline |
| POST | `/api/folders` | create a folder |
| PATCH/DELETE | `/api/folders/{id}` | rename or move / delete, leaving files at the root |
| DELETE | `/api/sources/{id}` | drops rows; blob goes when unreferenced |
| GET | `/api/sources/{id}/document` | markdown-latex rendition + chunks |
| POST | `/api/sources/{id}/reingest` | re-run the analyzer over stored bytes |
| POST | `/api/sources/{id}/ask` | `{"question": "..."}` to the analyzer, on the ORIGINAL file |
| GET | `/api/search?q=` | FTS5 over chunks (`source_id`, `limit` optional) |
| GET | `/c/{id}` | permalink; serves the app, which opens that conversation |
| GET | `/api/conversations` | the sidebar list, newest first |
| GET | `/api/conversations/{id}` | full transcript, compacted turns included |
| PATCH/DELETE | `/api/conversations/{id}` | rename / delete |
| POST | `/api/conversations/{id}/compact` | fold the transcript into a summary and clear the context |
| POST | `/api/chat` | `{"message", "conversation_id"?, "source_ids"?, "model_id"?, "effort"?}`; closing the connection cancels the provider call |
| GET | `/api/providers/presets` | the "Add Provider" menu |
| GET/POST | `/api/providers` | list / connect a provider (import runs on create) |
| PATCH/DELETE | `/api/providers/{id}` | edit name, URL or key / disconnect |
| POST | `/api/providers/{id}/refresh` | re-read the endpoint's catalogue |
| GET | `/api/models` | `?include_hidden=`, `?provider_id=` |
| POST | `/api/models` | add a model the endpoint doesn't advertise |
| PATCH/DELETE | `/api/models/{id}` | pin, hide, edit capabilities / remove |
| GET/PATCH | `/api/settings` | role assignments + thinking effort |

## Conversations

Everything you ask is saved against your user, so a reload puts you back in the
same conversation. The rail lists them all; click one to reopen its transcript.

When a conversation gets long, `/compact` in the composer summarises it: the
turns stay readable with a divider marking the fold, but the next question
carries the summary instead of the whole exchange, and the context meter drops
back to zero.

## Status

Phase 0 of AGENTS.md is in place: server, storage, schema, both provider wire
formats with tool-calling, multi-modal parts and reasoning effort, the per-user
model registry behind the prompt bar, retrieval, and an end-to-end text/markdown
ingestion path that needs no API key. Uploading a PDF, image, audio or video file
stores the original and marks the source `failed` with "analyzer not implemented
yet", which is phase 1.

There is no registration: the server seeds one local user and treats it as the
current one. Provider keys are stored in plaintext in SQLite, which is the same
trust level as the `.env` they replace. Encrypt them before this runs anywhere
multi-user.
