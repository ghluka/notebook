# 📚 notebook

An agent harness over your own sources: upload documents; an analyzer model
vectorizes each one into a database so a researcher can easily index sources
and ask questions with citations. It answers only from your sources, so it
knows only what you give it and avoids hallucinations. Supports vaults; you
can isolate and organize each source so it stays on the topic of that vault.

<img src="https://luka.onl/f/62742f1581b4.png" width="510">

[📽️ Video demonstration - Calculus prompt](https://luka.onl/f/1da1305e4940.mp4) (data: two >700 page [textbooks](https://openstax.org/subjects/math#Calculus))


## Running

> [!WARNING]  
> When you first run, you'll be asked to set a password. Do not expose the site
> beyond locally until this has been done.

```bash
cargo run
```

When you first run, make sure you do not publicly expose the site, as you'll want
to set a password first.
The default bind is localhost:8080, so open http://127.0.0.1:8080. Once it's set
up, click the model chip in the prompt bar and choose **Configure**, then add a
provider. Any API endpoint that supports the OpenAI-style responses or chat
completions, or any endpoint that supports the Anthropic-style messages endpoint
will work.

`.env` (see `.env.example`) is optional and covers server settings only: bind
address, database and upload paths.

### Behind a reverse proxy

If you wish to serve this from a subdirectory, you can set the path in .env by
setting the `BASE_PATH`.