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
- **reqwest** for provider HTTP. No vendor SDKs. The wire formats are small
  and we want to point at local/proxy endpoints by changing a base URL:
  Anthropic-style `/v1/messages`, and OpenAI-style in its two shapes.
  `/responses` is tried first (LM Studio, DeepSeek and OpenAI serve it, and
  NVIDIA for some of its models); `/chat/completions` is the fallback for
  endpoints without the route (Google's compatibility layer 404s it) and for
  audio and video, which have no Responses part. A 404, 405 or 501 on
  `/responses` means the route is missing, never a 429 or a validation error.
  Support is per model, not per endpoint: NVIDIA's gateway serves the route
  for its omni model and 404s it for lightning on the same base URL. So the
  answer is remembered per base URL and model for the life of the process,
  and the fallback costs one request per model, not one per turn. The
  fallback's log line carries the provider's reason.
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
  base.rs              the URL prefix when a proxy serves this from a subdirectory
  analyzer/            ingestion: bytes → markdown-latex → chunks
    mod.rs             dispatch on source kind
    text.rs            no-LLM path for text/markdown (works today)
  llm/
    types.rs           provider-neutral Message/ContentPart/Tool/ToolCall types
    openai.rs          OpenAI-style: /responses first, /chat/completions fallback
    responses.rs       the /responses translation (items, reasoning, stream events)
    anthropic.rs       Anthropic-style /v1/messages
    mod.rs             LlmProvider trait + registry
  routes/
    mod.rs             router assembly
    health.rs
    sources.rs         upload, organise, inspect, search; folders live here
    vaults.rs          list, create, rename, delete and open vaults
    chat.rs            researcher endpoint
    models.rs          providers, models, settings, the Configure panel
static/index.html      the whole UI: explorer, chat, viewer, Configure and Settings, vaults
```

## 4. Model configuration

A user connects a **provider** (a base URL + an API style + their API key). The
server reads that endpoint's own model list and imports every model it reports;
the user then **hides** the ones they never want to see and **pins** the ones
that should appear in the prompt-bar picker. Two of those models are assigned
roles: `researcher_model` and `analyzer_model`.

```
users ──< providers ──< models
   └──< settings (researcher_model, analyzer_model, thinking_effort, vault)
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
  the answer that open the cited source. The marks themselves come out of the
  prose at render time in whatever brackets the model used: some models cite in
  the lenticular brackets their training uses, with narrow no-break spaces
  inside, so `stripCitations` accepts those look-alikes and any kind of space,
  and the server's `plain_marks` folds them to plain brackets before matching,
  so such a mark still resolves to its exact line instead of a word-overlap
  guess. The stored answer keeps what the model wrote. A mark that dropped its
  file name after the first citation, `[lines 9-13]` or `[p. 4]`, is stripped
  by its locator alone, and marks are stripped before the maths is found, so
  one the model put inside a formula goes too. In the maths itself: `\text{}`
  keeps its spaces, cut from the source rather than rebuilt from tokens, with
  the ends padded because MathML trims them; `\mathbb`, `\mathcal`,
  `\mathfrak`, `\mathbf` and friends become the real Unicode letters (ℕ, 𝒜,
  𝔄, 𝐀), because browsers honour no `mathvariant` but normal; an inline span
  may not run through another `\(`; and a `\(` that never closes is closed at
  the end of its sentence, but only when what is there renders as LaTeX.
  Markdown tables render: a header row, a row of dashes with as many columns,
  then rows until a line without a pipe. Alignment colons are honoured, pipes
  inside a code span or escaped as `\|` do not split a cell, and because maths
  is taken out before blocks are parsed, `$|x|$` in a cell is safe. A table
  scrolls inside its own box, never the page. Narrow no-break spaces between
  words render as ordinary spaces, since some models use them everywhere and
  at a fifth of an em "is T" reads as "isT". Maths that is empty once the
  marks are out (a model that wraps its citation in `\[ \]`) draws nothing
  rather than an empty source box, and a line that is only a `[Source: ...]`
  note is stripped like a mark. A mark never contains another opening bracket
  and never starts at a bracket straight after a backslash, which is maths:
  otherwise `\[ [notes.pdf, line 3] \]` loses the maths' own bracket along with
  the citation and leaves `\ \]` behind as text.
- **What the model did is shown between the question and the answer.** Each
  stretch of reasoning and each tool call is one quiet line, in the order they
  happened: "Thought for 12 seconds", "Searched cardinality", "Read
  02SetsAndPropositions.pdf from line 1". The line has no box of its own and its
  chevron sits after the label; opening it shows the detail in a rounded card
  underneath: the reasoning itself, or what the call found (a search's files and
  lines, a read's line range, from `tool_found`). Reasoning streams on a channel
  of its own (`reasoning_content` or `reasoning` in chat completions,
  `response.reasoning_text.delta` or the summary delta in Responses,
  `thinking_delta` in Anthropic's format), becomes `StreamEvent::Thinking`, and
  reaches the page as a `thinking` event; a tool call is a `tool` event when it
  starts and `tool_done` when it returns. The stretch being written is open and
  counts its own seconds, and folds when anything else arrives. The server keeps
  the same steps (`TraceStep`, timed on its side) and stores them on the message
  (`trace`, migration 0010), and a finished turn is redrawn from that stored
  copy, so a live turn and a reopened one read identically. Answers from before
  traces keep their reasoning as a single step. Nothing in the trace is sent
  back to the model or searched, and tool calls no longer repeat in the meta
  line.
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
- **Vaults** are separate libraries, the way Obsidian has them. A vault owns
  its sources, folders and conversations (`vaults`, migration 0012, which puts
  everything that existed into one vault called Library). One is open at a
  time, the `vault` setting, read through `db::active_vault`, which falls back
  to the oldest vault when the setting points at a deleted one and makes one
  when there are none, so an upload always has somewhere to land. Every list
  and every search reads the open vault. A conversation keeps the vault it was
  started in: its search, catalogue and tools read that vault whichever one is
  open, and opening it from a link or the back button opens its vault too. The
  switcher sits at the foot of the rail under the status line, with the
  settings gear to its right; its menu lists the vaults and opens the manager,
  which creates, renames and deletes them. Move to vault, on a file (or the
  selection) or on a folder (with its subfolders and files), is how an existing
  library gets sorted; either lands at the target's root. Deleting a vault
  deletes what is in it, and a stored file goes only when no source in any
  vault still uses its bytes. The last vault cannot be deleted. Nothing
  selected, attached or open in the viewer survives a switch, and each vault
  remembers its own last conversation in `localStorage`.
- **Settings** (the gear) sets the researcher, the analyzer and the thinking
  effort, and links to Configure for providers and the full model table. The
  analyzer is offered only models tagged vision, plus whichever holds the role
  now, so the select never names a model that is not the one in use.
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
- **Attachments belong to the question they were sent with.** Drag files from
  the explorer into the conversation and they appear as chips above the
  composer. Sending takes them with the question (`source_ids` is exactly that
  list, captured at send time, so a question queued behind a running answer
  keeps its own files) and clears the composer for the next one. They show
  under the question as chips that open the file, and are stored with it
  (`messages.attachments`, migration 0011, as id, title and kind, so a file
  deleted later still has a name). A retry or an edit asks with the same files,
  since the rewind endpoint hands them back with the text, and an undo puts both
  the text and the files back in the composer. With nothing attached the search
  covers the whole library. This replaced attachments that persisted across
  turns, where nothing on the thread said which files an earlier question had
  asked about.
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

**A rendition is a transcription, and a description of the file is not one.**
Small models handed page images answer with a caption instead: fluent, wrong in
the details, and useless as a rendition, because the notebook then indexes a
description of the document in place of the document and the researcher quotes
it as if it were the source. Storing that is worse than failing.

- Every PDF path checks what came back. A rendition that follows the `## p. N`
  shape it was asked for is taken at its word. One that has no page headers and
  opens by naming the artefact it came from ("The PDF contains", "This document
  is", "The image shows"), or that runs to less than 250 characters per page it
  claims to cover, is a description. Where the input is known, as it is for a
  structured text layer, a rendition that kept less than 40% of what it was
  given is a summary however well it reads.
- The failure is its own error kind, `AppError::Rendition`, because it means the
  model reached the file and would not transcribe it. Asking the same model the
  same thing along another path only produces the same answer, so those paths
  are skipped, and the message names the model and says to pin a stronger one.
- **A text layer is the fallback, and the raw text layer is the last resort.**
  When the model will not transcribe a file that has a usable text layer, the
  extraction itself is stored as the rendition, under a note saying so. A plain
  rendition that is true beats a fluent one that is not: it is searchable,
  citable, and nothing in it was invented.
- **The text layer is read through the fonts.** Word, LaTeX and journal PDFs
  store text as glyph ids that mean nothing without the font's ToUnicode map,
  so scanning the file's raw strings finds gibberish, and for a while that
  judged nearly every real paper to have no text layer at all, leaving it at
  the mercy of the vision model. `render::extract_text` runs hayro's
  interpreter, the one that draws the page images, with a device that writes
  down each glyph's character instead of drawing it. Spacing comes from where
  the glyphs land: a lower baseline is a new line, a large drop or a jump back
  up to the next column is a new paragraph, a gap wider than a thin space is a
  word break. Ligatures are unfolded, since "classiﬁcation" with the ﬁ glyph
  is a different word to the search index. Invisible text, a scanned paper's
  OCR, counts. Each page sits under its own `## p. N` header, so a text-layer
  rendition cites by page like a page-image one. The raw string scan is kept
  only for files the interpreter cannot parse.
- **The analyzer reasons only when its model is tagged thinking.** The effort
  in the prompt bar is chosen for the researcher; a model without the tag
  gets none for transcription (`analyzer_effort`). An empty reply says why:
  the output limit reached (and whether it went on reasoning), reasoning with
  no text, or the stop reason the provider gave, rather than "returned no
  text", which sent people hunting for a model that can see.
- **A failed run leaves no description behind.** When a run fails this way and
  what is already stored is itself a description, that rendition and its chunks
  are dropped, so a source the explorer marks `failed` is not still answering
  questions from a caption. A stored rendition that reads as a real
  transcription stays: a failed rerun is no reason to throw away good work.

**An endpoint that refuses the kind of content is a configuration problem, and
the client is told so.** `supports_vision` is a claim someone typed; the
endpoint's answer is evidence. When a provider says a model does not take images
(NVIDIA: "does not support image inputs"; LM Studio, handed an OpenAI `file`
part: "'content' objects must have a 'type' field that is either 'text' or
'image_url'"), `LlmError::rejects_modality` recognises it and the analyzer stops
rather than working through fallbacks that ask the same model the same thing in
a different envelope.

- The failure is `AppError::Modality`, carrying the model it happened on, and it
  reaches the client as `kind: "modality"` the way a rate limit reaches it as
  `kind: "rate_limited"`. Both then ask the same question in the same dialog:
  which model should read this instead. Picking one reingests on it.
- Provider errors are quoted, not dumped. `provider_message()` digs the sentence
  out of `error.message`, `message` or `detail`, flattens it to one line and cuts
  it at 300 characters, so a failure reads as a sentence instead of as JSON
  nested inside three layers of our own prose.
- A file with a usable text layer falls back to it instead of failing, since the
  refusal is about images, not about the file.

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
- **A follow-up that names nothing is searched with the exchange before it.**
  "can you summarize it?" has one searchable word, so searching it alone finds
  junk, and a small model answers from whatever it is handed: it once
  summarised a textbook chapter on cardinality instead of the reading the
  previous answer had just named. `retrieval_query` adds the last answer and
  the last question to any question with fewer than two searchable words
  (`db::search_words`), with citation marks taken out so a file the
  conversation had moved away from is not pulled back in. Two searchable words
  are enough to stand alone: "what about the second reading?" is searched as
  it is, so the previous topic cannot drag it back. History is therefore read
  before the first search, not after. Every citation names its file and none
  goes inside mathematics, which the prompt says outright, because the renderer
  can strip a bare `[lines 9-13]` but cannot tell which file it meant.
- **Grounded is not the same as passive.** The prompt says facts come from the
  sources and nothing else, and a small model read that as a ban on making
  anything: asked to make a truth table, it answered "yes, you can make a truth
  table" and pointed at a file. So the prompt says outright that the working
  is the model's to do (an example, a table, a solved exercise, a proof), with
  the facts it rests on cited, and that a request to make something is met by
  making it. It also names the citation that breaks a sentence, "as shown in
  [file]", since the rule alone did not stop it.
- **A turn that works and then stops is asked once to answer.** Small models
  sometimes reason their way to a search and then write the call into their
  reasoning as text instead of making it, and the round ends with no text and
  no call: the trace showed four tool calls and then nothing. `stalled()` names
  that round (no answer, and no call or no budget to run one), and the loop
  drops its output and sends `ANSWER_NOW` once. The tools stay defined for that
  round, because a history with tool calls in it is refused by Anthropic's
  format when no tools are, and the model is told not to search instead. If it
  still says nothing, the page says so where the answer would be.
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
| `BASE_PATH` | none | URL prefix when a reverse proxy serves this from a subdirectory, such as `/notebook` |

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

### Behind a reverse proxy

Every route here is written against the site root, so a proxy that serves the app
from a subdirectory has to be told about. `BASE_PATH` names that prefix.

```nginx
location /notebook/ {
    proxy_pass http://127.0.0.1:8080;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_read_timeout 3600s;        # an analysis is minutes of model calls
    client_max_body_size 256m;       # nginx's own cap is 1m, under MAX_UPLOAD_BYTES
}
location = /notebook { return 308 /notebook/; }
```

- **`BASE_PATH=/notebook` works with either proxy style**: one that passes the
  prefix through (`proxy_pass http://127.0.0.1:8080;`) and one that strips it
  (`proxy_pass http://127.0.0.1:8080/;`). The router mounts the same table at
  the root and under the prefix, and the prefix is put back on every URL the app
  emits, so the redirect that sends an anonymous browser to the login page, the
  session cookie's `Path` and the client's own fetches all stay inside the
  subdirectory.
- **A proxy may declare the prefix instead**, with `X-Forwarded-Prefix:
  /notebook`, honoured when `BASE_PATH` is unset. That only works with the
  stripping style, since the router is built once at startup and cannot mount a
  prefix it only learns about per request.
- **The client learns the prefix from the page.** Each entry point carries a
  `__BASE_PATH__` placeholder in `<head>` that `routes::pages` fills in, and
  everything in `static/` builds its URLs as `BASE + '/api/...'`. Nothing there
  may hard code a leading `/`, or the subdirectory deployment breaks in a way
  the server cannot see.
- **The mount point without its trailing slash redirects to the one with it**,
  which is the spelling the client uses, rather than answering 401 through the
  gate and sending people looking for a login page that was already there.
