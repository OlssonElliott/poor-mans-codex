# Poor Man's Codex

A lightweight local Codex-style workflow that connects ChatGPT with your local Git repositories.

The installed CLI command is `chatcode`.

ChatCode does not use the OpenAI API. Instead, it packages relevant repository context into a file you upload to ChatGPT. ChatGPT returns a unified Git patch, and ChatCode validates, applies, tests, stores and reviews that patch locally.

## Features

- Builds task-specific context from the current Git repository
- Maintains a persistent, incrementally updated project graph
- Includes local staged and unstaged changes
- Filters common secret files and redacts common secret patterns
- Applies unified diffs with `git apply`
- Runs project tests automatically when a supported test setup is detected
- Stores patch history with before and after snapshots
- Can safely undo the latest ChatCode patch
- Opens historical changes as side-by-side diffs in VS Code
- Requires no OpenAI API key

## Requirements

- Python 3.11 or newer
- Git
- ChatGPT
- VS Code with the `code` command available in `PATH` for review support

Your target project also needs its normal test tooling installed if you want ChatCode to run tests automatically.

## Installation

Clone the repository:

```bash
git clone git@github.com:OlssonElliott/poor-mans-codex.git
cd poor-mans-codex
```

Install it in editable mode.

Windows:

```powershell
py -m pip install -e .
```

macOS or Linux:

```bash
python3 -m pip install -e .
```

Verify the installation:

```text
chatcode --help
```

The repository is named `poor-mans-codex`, but the CLI command is `chatcode`.

## Basic workflow

Run ChatCode from inside the Git repository you want ChatGPT to work on.

Create a task:

```text
chatcode patch-context "Add validation to the contact form"
```

ChatCode creates a repository-specific workspace and opens it automatically.

The two files you normally care about are:

```text
UPLOAD_TO_CHATGPT.md
patches/
    incoming.diff
```

Upload `UPLOAD_TO_CHATGPT.md` to ChatGPT.

The generated file contains the task, relevant source files, Git state and response instructions. ChatGPT should return one unified diff inside a single `diff` code block.

Copy the contents of that code block into:

```text
patches/incoming.diff
```

Then run:

```text
chatcode apply
```

ChatCode validates and applies the patch locally. It then runs tests when possible and asks:

```text
ChatCode: Do you want to review changes? (y/n):
```

Enter `y` to open the exact before and after change as a side-by-side diff in VS Code.

## Commands

### `chatcode status`

Show the current repository and Git status.

```text
chatcode status
```

### `chatcode context`

Generate a new ChatGPT context file for a task.

```text
chatcode context "Describe the task here"
```

Each new context clears `patches/incoming.diff` so an old patch is not accidentally reused.

### `chatcode patch-context`

Generate patch-oriented context from the exact current working-tree files:

```text
chatcode patch-context "Describe the code change here"
```

This is the preferred command when asking ChatGPT to produce a patch. Relevant files are read directly from disk; modified patch targets are preferentially included in full, while very large files use labeled, symbol-aware excerpts. Every included source file has a SHA-256 digest. Paths in the generated context use forward slashes.

General `chatcode context` behavior remains available for analysis tasks.

Both context commands synchronize `project-map.json` in the target repository's
ChatCode workspace. The compact index stores file hashes, summaries, tags,
high-level symbols, imports and direct source dependencies. It is not written
inside the target repository. Unchanged files are not reanalyzed, so edits made
manually in an IDE are detected on the next context command without requiring
`chatcode apply`.

Python is analyzed with its standard AST. JavaScript/TypeScript and the other
currently indexed code suffixes use a conservative structural fallback. File
metadata ranking and shallow dependency expansion supplement the existing
filename, content and Git-change ranking. Low-level calls and state accesses are
not persisted.

ChatCode has two explicit indexing modes. `ai` uses compact Qwen metadata as the
primary file selector and does not run the repository-wide legacy content
ranker. `static` never invokes Ollama and uses deterministic path, symbol,
import and content ranking. Both modes retain only lightweight hashes,
languages, symbols, imports and direct dependencies in `project-map.json`.

Select a mode in ChatCode's local `.env` (copy `.env.example`):

```powershell
$env:CHATCODE_QWEN_MODEL = "qwen2.5-coder:7b"
$env:CHATCODE_INDEX_MODE = "ai"
chatcode context "why does the door reopen?"
```

An explicit `CHATCODE_INDEX_MODE` always wins. If it is omitted, the older
`CHATCODE_QWEN_ENABLED` toggle remains supported; otherwise a configured
`CHATCODE_QWEN_MODEL` selects AI mode and no model selects static mode.

Before AI indexing starts, ChatCode invokes the configured model through the
same prompt, structured-output and parser path used for real files. A failed
preflight skips the semantic phase and uses static retrieval for that run. A
circuit breaker also stops a run when at least four of its first five file
analyses fail for the same reason. Completed semantic results remain cached.

Qwen only returns a compact summary, tags and important symbol names; it never
writes code or produces a relation graph. Missing Ollama, timeouts, model errors
and malformed responses are classified while deterministic metadata remains
available as a fallback.

The deterministic map is saved before Qwen starts. Semantic state is cached per
file using its source hash, model name and analyzer version, and each completed
file is checkpointed atomically. Context generation shows semantic progress and
an approximate ETA. If it is interrupted, rerunning the command resumes pending
files without repeating completed work. `chatcode apply` updates the static map
immediately and leaves semantic enrichment for the next context command.

### `chatcode apply`

Apply the default `patches/incoming.diff`.

```text
chatcode apply
```

Apply a specific patch file:

```text
chatcode apply path/to/change.diff
```

Skip automatic tests:

```text
chatcode apply --no-test
```

Skip the review prompt:

```text
chatcode apply --no-review
```

ChatCode also strips an outer Markdown `diff`, `patch` or unlabeled code fence from `incoming.diff` before applying it.

Before changing any file, `chatcode apply` parses the unified diff, recalculates hunk counts, serializes a canonical patch, asks Git to parse it with `git apply --numstat`, and then runs `git apply --check`. Only a patch that passes every stage is applied.

Malformed syntax, stale context, and valid patches that do not match the working tree are reported as separate failure types. Each produces `PATCH_REPAIR_CONTEXT.md` with failure-specific diagnostics and exact current working-tree context. A failed validation never partially applies the patch, and ChatCode never uses `git apply --reject`.

### `chatcode test`

Run the detected project test suite manually.

```text
chatcode test
```

ChatCode currently detects test setups for:

- npm, pnpm, Yarn and Bun
- Maven
- Gradle
- pytest
- Composer

If no supported test command is found, ChatCode reports that tests could not be run automatically.

### `chatcode undo`

Reverse the latest patch applied by ChatCode.

```text
chatcode undo
```

Undo without running tests afterwards:

```text
chatcode undo --no-test
```

Undo uses the stored patch and refuses to force a reversal when the current files no longer match safely.

### `chatcode history`

Show ChatCode patch history.

```text
chatcode history
```

History records whether an entry is applied or undone, the affected files and available test status.

### `chatcode review`

Review the latest ChatCode history entry in VS Code:

```text
chatcode review
```

Review another history entry by number:

```text
chatcode review 2
```

New history entries include before and after snapshots, so review shows the exact change ChatCode applied even if the working tree later changes.

## Workspace

Generated files are stored under ChatCode's own `workspace` directory rather than inside the repository being edited.

A workspace looks roughly like this:

```text
workspace/
    my-project/
        _<repo-hash>/
            UPLOAD_TO_CHATGPT.md
            patches/
                incoming.diff
            test-results/
                latest.md
            history/
                applied/
                undone/
```

`UPLOAD_TO_CHATGPT.md` is the request you upload to ChatGPT.

`patches/incoming.diff` is where you paste the patch returned by ChatGPT.

`test-results/latest.md` contains the latest detected test run.

`history/` stores applied and undone ChatCode changes.

## Safety

ChatCode performs several checks before generated changes are applied.

It validates patch paths, prevents patches from modifying `.git`, checks patches with `git apply --check`, and uses reversible Git patches instead of destructive commands such as `git reset --hard`.

Context generation ignores common secret files such as `.env`, credential files and private keys. It also redacts several common token, password and secret patterns from source context, Git diffs and test output.

Secret detection is best-effort and is not a guarantee. Review generated context before uploading it if a repository contains sensitive information.

## How the ChatGPT bridge works

ChatCode itself contains no AI model and does not automate the ChatGPT interface.

The workflow is deliberately simple:

```text
local repository
      |
      v
chatcode context
      |
      v
UPLOAD_TO_CHATGPT.md
      |
      v
ChatGPT
      |
      v
incoming.diff
      |
      v
chatcode apply
      |
      +--> tests
      +--> history
      +--> review
      +--> undo
```

This keeps your local Git repository under your control while still allowing ChatGPT to work with local, unpushed and uncommitted code.
