# notebook: agent and contributor guide

A local-first research harness: you upload sources (PDFs, images, video, audio,
text, URLs), an **analyzer** model turns each one into an indexable rendition,
and a **researcher** model answers questions over that index with citations,
the way a coding agent works over a repo, except the "repo" is your source
library.

This file is the contract for anyone (human or agent) working in this repo.
Read it before writing code.

---

## 1. The idea

Two models with different jobs:

| | analyzer | researcher |
|---|---|---|
| modality | multi-modal (vision + documents + audio transcripts) | text only |
| scope | one source at a time, in full | the whole library, through tools |
| job | turn bytes into **markdown-latex** + **chunks** + **metadata** | search, read, reason, answer, cite |
| invoked by | ingestion pipeline; also on demand by the researcher | the chat endpoint |
| cost profile | expensive, runs once per source | cheap per turn, runs constantly |

The key move: **the original file is never thrown away.** The analyzer produces
a lossy-but-searchable rendition for fast retrieval, and when the researcher
hits the limits of that rendition it re-invokes the analyzer as a sub-agent
against the original bytes: "look at figure 3 on page 12 and tell me the axis
units". That's the `ask_source` tool.

```
upload ──► storage (uploads/<sha256>) ──► analyzer ──┬─► documents.markdown  (markdown-latex, human+LLM readable)
                                                     ├─► chunks + chunks_fts (retrieval surface)
                                                     └─► sources.metadata    (structured facts)

chat ──► researcher ──► tools ──┬─ search_sources(query)     → FTS5 over chunks
                                ├─ read_document(source_id)  → full markdown, or a chunk range
                                ├─ list_sources()            → titles, kinds, summaries
                                └─ ask_source(source_id, q)  → re-invokes the ANALYZER on the original file
```

## 2. Stack (decided; do not swap without a reason in the PR body)

- **Rust**, edition 2024, single binary.
- **axum 0.8** for HTTP (path params use `{id}`, not `:id`).
- **SQLite via sqlx 0.9**, runtime queries (`sqlx::query`), *not* the compile-time
  macros, because the build must not require a live DATABASE_URL.
- **FTS5** for retrieval in v1. Vectors are a later addition *behind the same
  `search_sources` tool*, so the researcher never learns which is in use.
- **reqwest** for provider HTTP. No vendor SDKs. The two wire formats
  (OpenAI-style and Anthropic-style) are small and we want to point at
  local/proxy endpoints by changing a base URL.
- **Model configuration lives in the database, not the environment** (§4).
- Files live on disk under `uploads/`, addressed by sha256. The DB stores paths,
  never blobs.

## 3. Layout

```
migrations/            sqlx migrations, applied at startup
src/
  main.rs              wiring: config → db → state → router → serve
  config.rs            env parsing; provider + model selection
  error.rs             AppError → HTTP response; every handler returns AppResult
  state.rs             AppState { db, storage, llm registry, config }
  storage.rs           content-addressed file store (uploads/)
  db.rs                pool setup, migrations, row structs
  models.rs            registry: users, providers, models, role assignments
  analyzer/            ingestion: bytes → markdown-latex → chunks
    mod.rs             dispatch on source kind
    text.rs            no-LLM path for text/markdown (works today)
  llm/
    types.rs           provider-neutral Message/ContentPart/Tool/ToolCall types
    openai.rs          OpenAI-style /v1/chat/completions
    anthropic.rs       Anthropic-style /v1/messages
    mod.rs             LlmProvider trait + registry
  routes/
    mod.rs             router assembly
    health.rs
    sources.rs         upload, organise, inspect, search; folders live here
    chat.rs            researcher endpoint
    models.rs          providers, models, settings, the Configure panel
static/index.html      the whole UI: explorer, chat, viewer, Configure panel
```

## 4. Model configuration

A user connects a **provider** (a base URL + an API style + their API key). The
server reads that endpoint's own model list and imports every model it reports;
the user then **hides** the ones they never want to see and **pins** the ones
that should appear in the prompt-bar picker. Two of those models are assigned
roles: `researcher_model` and `analyzer_model`.

```
users ──< providers ──< models
   └──< settings (researcher_model, analyzer_model, thinking_effort)
```

Rules that matter:

- **The database is the source of truth.** The environment is read once, to seed
  an empty database, and ignored forever after. Do not add new behaviour that
  reads provider config from env at request time.
- **Keys never come back out.** `Provider.api_key` is `skip_serializing`; the
  API returns `key_hint` ("••••abcd") and `has_key`. A PATCH without `api_key`
  keeps the stored one. Keys are stored in plaintext in SQLite, the same trust
  level as the `.env` they replaced, and the thing to fix before this is
  multi-user for real.
- **Everything is owned.** Providers carry `owner_id`; models and settings hang
  off them. Registration does not exist yet: `bootstrap` creates one local user
  and `AppState::user_id` is it. Adding auth means filling that field from a
  session. Every query already scopes by it, so nothing else moves.
- **A missing key is only an error for remote endpoints.** Loopback providers
  (LM Studio, Ollama, vLLM) legitimately have none; see `models::is_loopback`.
- **Re-importing never clobbers user choices.** `import_models` refreshes the
  provider's facts and leaves `hidden` / `pinned` alone, and models the endpoint
  stopped reporting are kept with a stale `last_seen_at` rather than deleted.
- **What the endpoint says beats what the id suggests.** The OpenAI-compatible
  `/models` listing is often just ids (LM Studio and NVIDIA both are), so
  `fetch_remote_models` falls back to the server's own richer listing before it
  guesses: LM Studio's `/api/v0/models` gives `type: "vlm"`, `capabilities` and
  the real `max_context_length`, and OpenRouter gives `architecture`,
  `supported_parameters` and `context_length`. Guessing from the id
  (`Capabilities::guess`) is the last resort, and every flag is editable by
  clicking the tag in Configure. Flags gate two things: the context meter, and
  whether a model can hold the analyzer role.
- **Not every endpoint lists its models.** DeepSeek's Anthropic-style API lists
  none at all, and preview ids are routinely missing from endpoints that do
  list. So a model can be typed in by hand (`POST /api/models`, source
  `manual`) and every field of an existing one edited, the wire id included,
  since a typo should not cost the row its role and pin. Adding a provider that
  imports nothing opens that form straight away rather than leaving a dead end.
- **Non-chat models are imported hidden.** Embedding and reranking ids stay in
  the table so a refresh does not keep re-adding them, but they never reach the
  picker.

## 5. Interface

Three panes, in the NotebookLM arrangement: explorer, conversation, viewer.

- **Explorer** is a VS Code style tree: one line per entry, folders nest,
  right click opens a context menu, files carry a per-kind icon, and dragging a
  file onto a folder moves it. Selection follows file manager rules: click picks
  one, ctrl or cmd adds and removes, shift takes the range from the last click,
  and clicking empty space clears. Selection is only selection: what an action
  applies to and what a drag carries, never what the researcher reads. Every
  selected row is lit and the last clicked carries the brighter focus. A
  drag carries the whole selection, shows what is moving under the cursor, and
  scrolls the list when the pointer nears either edge, since a drag holds the
  pointer and the wheel does nothing.
- **Rows are patched, not rebuilt.** `renderTree` computes the shape of the tree
  first and only replaces the DOM when that shape changes; otherwise it updates
  the existing rows. Rebuilding on every 1.5s poll restarted every spinner
  mid rotation, which reads as an animation that skips. Folders are organisation only (`folders` table),
  and deleting one leaves its files at the root. Clicking a file toggles it into
  double click opens it in the viewer, and dragging it into the conversation
  attaches it to the next question.
- **Conversation** distinguishes the two speakers the way a chat should: your
  turn is a bubble on the right, the answer is full width prose with markdown
  rendered by the small renderer in `markdown()`. Maths is real maths:
  `texToMathml()` converts a LaTeX subset to MathML, which browsers lay out
  natively, so formulas need no library, no web fonts and no build step.
  Multi line mathematics is covered: `align`, `gather`, `cases`, the matrix
  family and friends become MathML tables, with `align` putting its odd columns
  right and even columns left so the equals signs stack the way TeX sets them.
  An environment written with no delimiters around it is still treated as
  display maths, since models emit them bare. An unknown macro that takes no
  argument renders as its own name, so one unrecognised symbol costs a symbol
  rather than sending a whole derivation back to raw source; a macro with
  arguments still fails the expression, because guessing its layout would be
  inventing mathematics. Only then is the source shown, which is never worse
  than what came before. `$...$` needs non-space at both ends so prices are not mistaken for
  maths. Citations become pills under
  the answer that open the cited source.
- **Turn actions** appear only while a turn is under the pointer, so a finished
  transcript stays quiet, and they never move anything when they appear: the bar
  always holds its space and only changes opacity. An answer carries copy and
  retry at the right of its meta line, next to when it was said in the shape a
  person would say it. A question carries copy, edit and undo under its bubble.
  Copy takes an answer as markdown with the citation marks already stripped, the
  same text that was rendered, and a question as it was typed.
- **Retry, edit and undo are one operation with three endings.** `POST
  /api/conversations/{id}/rewind` takes a question back out of the stored
  conversation and hands it to the client, which then asks it again unchanged
  (retry), asks a changed version (edit), or lets it go and puts the text back
  in the composer (undo), since an undo should hand you what you had rather than
  swallow it. The message id is either end of the exchange: an answer rewinds
  the question behind it, a question rewinds itself. The model never sees its
  own first attempt, which is the whole point of retrying. Everything after the
  rewind point goes too, because a conversation is a line, not a tree, and
  orphaned turns would make the history a lie. The answer to the question in
  hand goes without asking, since replacing it is what was asked for; anything
  beyond it earns a dialog naming how many turns will go. A compacted turn
  offers none of this, since its summary would no longer match.
- **Editing happens where the question is**, not in the composer at the bottom:
  the bubble becomes a box holding the text, Enter sends, Escape and Cancel put
  the bubble back untouched, and sending an unchanged question just closes the
  editor rather than burning a call.
- **Rail** is two equal halves, chats over sources, each scrolling on its own so
  neither can squeeze the other no matter how many rows it holds. Both carry the
  same section header: uppercase, faint, with its actions on the right.
- **Chats** live in the top half of the rail: every conversation this user has
  had, newest first, with a dot on the compacted ones. Each row has a delete
  button on hover and a right click menu (open, rename, compact, delete).
  The address bar is the permalink: opening a conversation pushes `/c/<id>`,
  a new one is `/`, and both browser history and a pasted link work because
  `/c/{id}` serves the same page and the client reads the id from the path.
  `localStorage` remembers the last one for a bare `/`.
- **Commands** start with `/` in the composer and complete as you type (arrow
  keys, Tab or Enter to run). There is exactly one so far, `/compact`: it folds
  the conversation into a summary, marks those turns compacted, and drops the
  context back to nothing. The transcript still reads in full, with a divider
  where the fold happened, but the next prompt carries the summary instead of
  the turns.
- **Context meter** reads the last turn's usage, which is stored per assistant
  message (`input_tokens`, `output_tokens`), so reopening a conversation shows
  what it actually cost instead of resetting to zero. Compaction sets it back to
  zero because that is the truth after a compaction.
- **Dialogs** are in-app, never native. `dialog()` returns a promise, and the
  three wrappers over it replace what the browser offers: `say()` for alert,
  `ask()` for confirm, `askText()` for prompt, plus `askChoice()` for picking
  from a list. Escape and a backdrop click cancel, Enter confirms, and a
  destructive dialog never puts focus on its own danger button. Do not
  reintroduce `alert`, `confirm` or `prompt`.
- **Attachments decide what gets searched.** Drag files from the explorer into
  the conversation and they appear as chips above the composer; `source_ids` on
  the next question is exactly that list. They persist across turns, since a
  line of questioning usually keeps its focus, and clear when the conversation
  changes. With nothing attached the search covers the whole library. This
  replaced an explorer selection that doubled as retrieval scope, where the
  thing being searched was invisible from the conversation itself.
- **Composer** runs one turn at a time. While a turn is in flight the send
  button becomes stop (Esc also stops), aborting the fetch; axum drops the
  handler when the connection closes, which cancels the outbound provider call,
  and since `/api/chat` persists only after a successful response, a stopped
  exchange leaves no trace in the conversation. Anything typed meanwhile is
  queued and sent when the turn finishes; "Ask now" on a queued item, or
  Ctrl+Enter, steers instead: it stops the running turn and jumps that question
  to the front.
- **Viewer** shows one source three ways: `Original` (the stored bytes, served
  by `/api/sources/{id}/raw`, so PDFs and images render natively), `Rendition`
  (the analyzer's markdown and its chunk count) and `Details`.

## 6. A file is what its bytes say it is

A name is a claim, not evidence. Every upload is identified in `analyzer::sniff`
from its leading bytes, and that answer, not the extension or the browser's
declared media type, decides how the file is analyzed. An mp3 renamed to `.pdf`
is analyzed as audio; a PDF renamed to `.mp3` is analyzed as a PDF. The stored
`kind` and `media_type` are the corrected ones, so the explorer icon and the
viewer's details panel show what the file really is next to the name it came
with.

- **Formats that cannot be read are refused at upload**, before they take up a
  slot in the queue, and the refusal names what the file actually is: "this file
  is a Word document", "a ZIP archive", "a Windows executable". One bad file in a
  drop of twenty never sinks the other nineteen: `POST /api/sources` answers with
  `uploaded` and `rejected` side by side, and the client shows one line per
  refused file.
- **The analyzer re-identifies from the stored bytes** rather than trusting the
  row, so files stored before any of this existed are handled correctly too, and
  a reingest cannot be fooled by stale metadata. Only the first 8 KiB is read for
  this, so identifying a two hundred megabyte video costs nothing.
- **Text is decoded as it was saved.** UTF-8 is nearly everything, but a byte
  order mark is honoured (UTF-16 either way), and a file that is not UTF-8 but
  reads as text in an older single byte encoding keeps its accents rather than
  being turned away as binary. NUL bytes and a scattering of control codes are
  what binary residue looks like, and that is what the test measures.
- **A file may not cost unbounded work.** Every ceiling below refuses loudly
  rather than truncating silently, because a document that is analyzed at all is
  analyzed in full:
  - one PDF stream may not inflate past 16 MiB, and all of them together not past
    64 MiB, which is what stops a kilobyte of crafted Flate from becoming a
    gigabyte of memory;
  - a PDF claiming more than 5000 pages is refused, since every page is a model
    call;
  - an image whose header claims more than 120 megapixels is refused before any
    model call, which is the decompression bomb that is small on disk;
  - text is read whole, so it stops at 32 MiB;
  - the per-kind byte ceilings (`IMAGE_MAX_BYTES` and friends) still apply, under
    the hard `MAX_UPLOAD_BYTES`.
- **A corrupt file fails honestly.** Rendering is already isolated per page, so a
  damaged PDF yields no pages instead of panicking, and the analyzer falls
  through to the native document and then to the text layer. When nothing can be
  read, the source is marked `failed` with the reason, and nothing is stored as
  if it were a rendition.

## 7. Ingestion is a queue

Storing a file and reading it are separate jobs, because they take wildly
different amounts of time: a local write is instant, a scanned PDF is minutes of
model calls.

- `POST /api/sources` writes the bytes, inserts the rows as `pending` and
  returns. `analyzer::spawn_analysis` takes it from there, so the work outlives
  the request. Refreshing the page or closing the tab no longer abandons
  whatever was still queued.
- One analysis runs at a time, process wide, held by the `analysis` semaphore in
  `AppState`. Analyzing a dropped folder in parallel is the surest way to get
  rate limited by your own provider.
- Status is the contract: `pending` (stored, waiting), `analyzing`, then `ready`
  or `failed`. The explorer greys the row and spins while a source is in either
  working state, shows progress in the rail footer, and polls `/api/sources`
  every 1.5s only while something is in flight.
- A failure stores `error_detail`, the same JSON the API would have returned, so
  the client can offer a model switch for a rate limit without re-running
  anything.
- `resume_pending` at startup picks up whatever a previous run left `pending` or
  `analyzing`, so a restart mid-queue is not a lost queue.

## 8. Rate limits

Providers refuse work in several dialects: a 429, a 503 with
`ResourceExhausted`, a 503 saying "high demand", an `overloaded_error`. They all
mean the same thing, and all of them pass if you wait.

- Every provider call goes through `llm::chat_with_retry`, which waits
  `RETRY_WAITS` (5s, then 60s, then 60s) between attempts, honouring a
  `Retry-After` header when one arrives, capped at 120s. Classification is by
  status and by body text, since not every provider uses a 429.
- Still refused after the last wait becomes `LlmError::RateLimited`, which
  handlers turn into `AppError::RateLimited` carrying the model it happened on.
  That responds 429 with `kind: "rate_limited"`, `model_id`, `model_name` and
  `waited_seconds`, which is how the client knows to offer a different model
  rather than reporting the file as unreadable.
- **A long analysis survives a restart.** Each finished batch is written to
  `sources.partial` with the page it reached, so a resume continues from there
  instead of re-reading the book. `ingest_with` clears it when the run is a
  deliberate fresh start on a hand picked model.
- **Every page gets read.** There is no page cap: `analyze_page_images` walks
  the whole document in batches of `PAGES_PER_REQUEST`, rendering each batch
  just before it is sent so a long book never holds more than a few page images
  in memory. A context window is a limit per request, not per document, so a
  long file becomes many requests joined into one rendition, chunked and indexed
  as a whole. Truncating and telling the reader to split the file is not an
  option: analyzing means analyzing. Pages that will not rasterize are named in
  the rendition rather than silently dropped, and `sources.progress` carries
  "done/total" so the explorer can show where it is.

- **A rate limit must never trigger a format fallback.** The PDF paths (page
  images, then the native document, then the text layer) exist for endpoints
  that cannot read a format. They all call the same endpoint, so retrying them
  against an exhausted provider only produces a slower, wronger error. Each
  fallback checks `is_rate_limited()` first and returns.
- Waiting is the server's job; choosing is the person's. After the schedule
  runs out, the client shows what happened and offers the other vision-capable
  models. `POST /api/sources/{id}/reingest` takes `{"model_id": "..."}` for
  exactly that retry.

## 9. Conventions

- Handlers return `AppResult<T>`; never `unwrap()` on request-derived data.
- All ids are UUID v4 strings. All timestamps are RFC3339 strings in UTC.
- Provider calls take `ChatRequest` and return `ChatResponse`. A handler must
  never construct provider-specific JSON. If a provider needs a field we don't
  model, add it to the neutral types first.
- Anything that costs money or takes >1s belongs behind a spawned task with a
  status column, not inside a request handler.
- New tables get a new numbered migration. Never edit an applied migration.
- `cargo build` must pass with no network and no API keys set.
- **The researcher has tools and is expected to use them.** One question is a
  short loop (`MAX_TOOL_ROUNDS`): `search_sources` runs another keyword search,
  `read_source` reads a rendition around a line with line numbers, and
  `list_sources` says what is in the notebook. The prompt tells it that the
  excerpts it was handed came from one keyword search on the user's own words,
  which is a starting point and not a verdict, and that it should read around a
  citation before concluding something is missing. A model without tool support
  gets the single-shot path.
- **Excerpts are whole passages, not snippets.** Search returns the full chunk
  plus its neighbours (`NEIGHBOUR_RADIUS`), because a worked example usually
  sits next to the sentence that names it, and a 32 token FTS snippet with
  ellipses is unusable as evidence. How much goes in is scaled from the model's
  own context window: a million token window gets hundreds of thousands of
  characters, not the same handful of fragments as a small one.
- **Attribution has a ladder.** Labels the model wrote win. Failing that,
  passages the answer visibly reuses (`echoed_hits`, several distinctive tokens
  in common) are credited, then files it deliberately opened. An answer that
  reuses nothing gets no citations, which is what keeps a "found nothing" reply
  from wearing eight source chips.
- **The researcher answers two kinds of message.** A question about the sources
  is answered from the excerpts and cited. A question about the conversation
  ("what did I just ask", "why that file?", "explain that again") is answered
  from the conversation, with no citation, and the excerpts are ignored. Without
  that distinction a model reports that your own question is missing from the
  excerpts, which is what it did. It must never guess which file would have held
  a missing answer either: naming one it cannot see is a fabrication.
- **Citations are what the answer used, not what was retrieved.** `cited_hits`
  keeps only the excerpts whose label appears in the answer, so an answer that
  found nothing carries no source pills. `excerpts_searched` still reports how
  many were looked at.
- **Retrieval drops function words** (`STOPWORDS` in `db.rs`). A long question
  otherwise buries its one rare term under chunks matching "give", "explain" and
  "depth". If a question is nothing but function words, they are searched
  anyway rather than returning nothing.
- The researcher prompt has two rules that exist because models broke them:
  cite by copying the bracketed label printed above each excerpt
  (`[08NumberTheoryII.pdf, p. 12]`), and open with the answer, never with
  "From the provided excerpts". The excerpt block is formatted to teach the
  citation format by example; it used XML attributes once and models echoed
  `source_title="..."` into their prose. Do not reintroduce key=value syntax
  there.
- No em dashes anywhere: code, comments, docs, or UI strings. Use a comma, a
  colon, a semicolon, or a full stop.
- The UI is one static file with no build step and no CDN. Keep it that way:
  vanilla DOM calls, no framework, no bundler.

## 10. Roadmap

**Phase 0: skeleton (done)**
- [x] axum server, config, error type, graceful shutdown
- [x] SQLite + migrations for sources / documents / chunks / FTS5 / conversations
- [x] content-addressed upload storage
- [x] provider-neutral LLM layer: OpenAI-style + Anthropic-style, incl. images,
      documents, tool-calling and reasoning effort
- [x] model registry: per-user providers with stored keys, catalogue import from
      the endpoint's own model list, hide/pin, role assignment
- [x] prompt bar: model picker, thinking effort, context-window meter
- [x] explorer with folders, icons, context menus and drag to move; multi-file
      upload; source viewer for original bytes, rendition and details
- [x] stop and queue: abort a running turn, queue what you type meanwhile, steer
      to interrupt and jump the queue
- [x] saved conversations per user, reopened from the sidebar and restored on
      reload; `/compact` summarises a conversation and clears its context
- [x] upload / list / get / delete / search endpoints
- [x] text + markdown ingestion end-to-end, no LLM required
- [x] one-shot `/api/chat` against the researcher model

**Phase 1: the analyzer**
- [ ] PDF: page images + text layer → per-page markdown-latex with `p. N` locators
- [ ] Images: description + OCR + embedded-LaTeX transcription
- [ ] Audio/video: transcription with `HH:MM:SS` locators; keyframes for video
- [ ] Structured `metadata` extraction (authors, date, entities, claim list)
- [ ] Job queue + status transitions (`pending → analyzing → ready|failed`) with
      retry and partial-progress persistence
- [ ] Chunking that respects headings, tables and math blocks; token estimates

**Phase 2: the researcher agent loop** (largely done)
- [ ] Tool loop over `search_sources` / `read_document` / `list_sources` / `ask_source`
- [ ] `ask_source` re-invokes the analyzer on original bytes, result cached per (source, question)
- [ ] Citations as `(source_id, chunk_id, locator)`, rendered as footnotes
- [ ] Context assembly beyond a flat replay: relevance-ordered prior turns
- [x] SSE streaming of tokens and tool events to the client (`POST /api/chat/stream`; `/api/chat` stays as the one-shot equivalent)

**Phase 3: retrieval quality**
- [ ] Hybrid search: FTS5 + embeddings (sqlite-vec), reciprocal-rank fusion
- [ ] Query rewriting/expansion by a cheap model before searching
- [ ] Cross-source entity linking; "what does source A say that B contradicts"
- [ ] Recency/authority weighting

**Phase 4: surface**
- [x] Three pane layout (explorer / chat / viewer)
- [ ] Studio pane: generated study artifacts (summaries, timelines, quizzes)
- [ ] Extend the LaTeX subset: environments, matrices, aligned equations
- [ ] Per-source notes
- [ ] URL ingestion + crawl depth 1
- [ ] Export a notebook (sources + renditions + chat) as a single archive

**Phase 5: operations**
- [ ] Registration and sessions; fill `AppState::user_id` from the session
      rather than the seeded local user, and encrypt `providers.api_key`
- [ ] Per-notebook isolation
- [ ] Token accounting and per-notebook budgets
- [ ] Prompt/response trace log for debugging agent behaviour
- [ ] Eval set: fixed sources + graded questions, run in CI

## 11. Environment

`.env` is optional and holds server settings only. API keys do not belong in it:
providers, keys, model visibility and role assignments are configured in the UI
and stored per user in the database.

The one exception is seeding a headless install. On a database with no providers
yet, `bootstrap` reads the seed-only variables below to create a provider and
assign the roles, once. After that the environment is ignored, and anything
configured in the UI wins.

| var | default | meaning |
|---|---|---|
| `BIND_ADDR` | `127.0.0.1:8080` | listen address |
| `DATABASE_URL` | `sqlite://data.db` | SQLite file (created if missing) |
| `UPLOAD_DIR` | `uploads` | content-addressed file store |
| `MAX_UPLOAD_BYTES` | `268435456` | 256 MiB request-body cap |

Seed-only (used once, against an empty database):

| var | default | meaning |
|---|---|---|
| `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL` | none / `https://api.anthropic.com` | seeds an Anthropic-style provider |
| `OPENAI_API_KEY`, `OPENAI_BASE_URL` | none / `https://api.openai.com/v1` | seeds an OpenAI-style provider |
| `RESEARCHER_PROVIDER` / `ANALYZER_PROVIDER` | `anthropic` | which seeded provider gets the role |
| `RESEARCHER_MODEL` / `ANALYZER_MODEL` | see `config.rs` | model id to create and assign |

Note that `dotenvy` does not override variables already exported in your shell.
if a provider looks wrong at first boot, check the real environment before the
`.env`.
