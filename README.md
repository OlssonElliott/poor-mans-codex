# Poor Man's Codex

A local, patch-based coding workflow that lets ChatGPT work against the **current state of a local Git repository** without giving ChatGPT direct filesystem access and without using the OpenAI API.

The installed CLI command is:

```bash
chatcode
```

ChatCode builds a task-specific context from your repository, uses a local Qwen model through Ollama to improve retrieval when AI indexing is enabled, packages the exact source ChatGPT needs into `UPLOAD_TO_CHATGPT.md`, and then safely consumes the unified diff ChatGPT returns.

The core workflow is:

```text
local repository
      │
      ▼
chatcode context "task"
      │
      ├── static project index
      ├── optional local Qwen semantic index
      ├── task/retrieval analysis
      ├── dependency + implementation expansion
      └── source/context contract
      │
      ▼
UPLOAD_TO_CHATGPT.md
      │
      ▼
ChatGPT
      │
      ▼
unified diff
      │
      ▼
patches/incoming.diff
      │
      ▼
chatcode apply
      │
      ├── patch validation
      ├── review/confirmation
      ├── pre-patch test baseline
      ├── relevant tests
      ├── full test suite
      ├── regression comparison
      ├── history
      ├── repair context on failure
      └── follow-up context if the user says the fix did not solve the problem
```

## What ChatCode is trying to solve

Large language models are usually good at modifying code **when they receive the right code**.

The difficult part is deciding:

- which files matter,
- which functions/classes inside those files matter,
- which dependencies have to be included,
- which backend/frontend/test surfaces belong to the same change,
- and how to provide enough code to patch safely without sending the entire repository.

ChatCode treats that context-building problem as a first-class part of the tool.

It does not simply search for filenames and dump them into a prompt. It maintains a compact project index, combines deterministic structure with optional local semantic analysis, expands likely implementation relationships, and finally materializes exact current source from the working tree.

## Requirements

Required:

- Python 3.11+
- Git
- ChatGPT
- the normal build/test tooling used by the repository you want to modify

Recommended:

- Ollama
- Qwen 2.5 Coder
- VS Code with the `code` command in `PATH` for side-by-side review

ChatCode can run in `static` mode without Ollama, but `ai` mode is the intended full retrieval workflow.

No OpenAI API key is required.

## Quick start

### 1. Clone and install ChatCode

Using HTTPS:

```bash
git clone https://github.com/OlssonElliott/poor-mans-codex.git
cd poor-mans-codex
```

Windows:

```powershell
py -m pip install -e .
```

macOS/Linux:

```bash
python3 -m pip install -e .
```

Verify:

```bash
chatcode --help
```

### 2. Install Ollama and Qwen

Install Ollama from:

```text
https://ollama.com/download
```

Then pull the default lightweight model:

```bash
ollama pull qwen2.5-coder:1.5b
```

If your hardware can comfortably run a larger model, you can configure another Qwen model instead.

### 3. Configure AI indexing

From the `poor-mans-codex` directory:

Windows:

```powershell
Copy-Item .env.example .env
```

macOS/Linux:

```bash
cp .env.example .env
```

The default configuration is equivalent to:

```env
CHATCODE_INDEX_MODE=ai
CHATCODE_QWEN_MODEL=qwen2.5-coder:1.5b
CHATCODE_QWEN_ENABLED=true
```

`CHATCODE_INDEX_MODE=ai` enables local semantic indexing through Ollama.

To disable model calls completely:

```env
CHATCODE_INDEX_MODE=static
```

### 4. Go to the project you want to modify

ChatCode is run **inside the target Git repository**, not inside the ChatCode repository.

Example:

```bash
cd C:\repos\my-project
```

### 5. Create context for a task

```bash
chatcode context "Add a new API endpoint for updating user profile settings."
```

ChatCode builds or updates the repository index, retrieves the relevant implementation surfaces and creates:

```text
UPLOAD_TO_CHATGPT.md
```

inside that repository's ChatCode workspace.

The workspace directory is printed and opened automatically.

### 6. Upload the generated file to ChatGPT

Upload `UPLOAD_TO_CHATGPT.md` as-is.

The file already contains:

- the task,
- current repository state,
- selected source,
- source line ranges,
- patch instructions,
- and the response format ChatCode expects.

For code changes, ChatGPT should return one unified diff.

### 7. Save the returned diff

Put the returned patch into:

```text
patches/incoming.diff
```

You can paste either the raw unified diff or the complete outer Markdown `diff` code block. ChatCode strips a supported outer code fence automatically.

### 8. Apply and validate

```bash
chatcode apply
```

ChatCode first checks the patch without changing the repository. In the interactive flow you can review the full diff before confirming the apply.

After the patch is applied, ChatCode runs relevant tests when it can safely identify them and then runs the full detected test suite.

If validation passes, ChatCode asks whether the patch actually solved your problem.

If you answer **no**, it asks what is still wrong and creates a fresh `FOLLOWUP_CONTEXT.md` against the new working-tree state.

---

# Most useful commands

| Command | What it is for |
|---|---|
| `chatcode context "task"` | Normal workflow. Builds a rich task context including selected source, repository structure, Git state and available test information. |
| `chatcode patch-context "task"` | Leaner patch-oriented context focused on exact current working-tree source. |
| `chatcode apply` | Validate, preview, apply and test `patches/incoming.diff`. |
| `chatcode check` | Run a standalone repository health check without first creating a coding task. |
| `chatcode repair` | Re-open the current freshness-validated repair context after a failed patch/test flow. |
| `chatcode followup` | Regenerate the latest unresolved follow-up context from the current repository state. |
| `chatcode undo` | Safely reverse the latest ChatCode-applied patch. |
| `chatcode review` | Open the latest ChatCode change as a before/after VS Code diff. |
| `chatcode review 2` | Review another history entry by number. |
| `chatcode history` | Show applied/undone ChatCode patch history. |
| `chatcode test` | Run the detected project test suite directly. |
| `chatcode status` | Show repository/branch status. |
| `chatcode status --reindex` | Throw away the cached project index and rebuild it from source. |

## `chatcode context`

```bash
chatcode context "Describe the change you want"
```

This is the most useful default command.

It includes more repository-level evidence than `patch-context`, including project structure, staged/unstaged Git state and the latest failed test output when available.

It also uses the same current-source materialization pipeline used for patch generation.

Each new context clears the default `incoming.diff` so an old patch cannot accidentally be reused for a new task.

## `chatcode patch-context`

```bash
chatcode patch-context "Describe the patch you want"
```

Use this when you want a smaller, explicitly patch-focused upload.

It still runs the retrieval/indexing pipeline and materializes exact current source, but avoids some of the extra repository context included by the normal `context` command.

## `chatcode apply`

```bash
chatcode apply
```

This is not a blind `git apply`.

Before repository files are changed, ChatCode:

1. normalizes the returned unified diff,
2. validates patch paths,
3. blocks absolute paths, path traversal and `.git` modifications,
4. validates hunk context,
5. asks Git to parse the patch,
6. checks whether the context became stale,
7. runs `git apply --check`,
8. creates a non-mutating preview.

In the normal interactive workflow ChatCode then shows a summary, offers a full diff review and asks for confirmation before modifying the repository.

After applying, it runs test validation and compares the result against the pre-patch baseline.

## `chatcode check`

```bash
chatcode check
```

Runs the project's detected test suite against the current working tree.

If stable failing test IDs are available, ChatCode can create a focused `CHECK_REPAIR_CONTEXT.md` for selected failures.

This is useful when the repository is already broken before you start a new ChatCode task.

## `chatcode repair`

```bash
chatcode repair
```

When a patch introduces a regression or a repair attempt still fails its target tests, ChatCode creates a repair context containing:

- the original task,
- the unsuccessful patch,
- failing test information,
- exact current working-tree source,
- and explicit repair targets.

`chatcode repair` only exposes that context if its working-tree assumptions are still fresh.

## `chatcode followup`

A patch can be technically correct and have green tests while still not solving the real problem.

After a successful apply ChatCode asks:

```text
Did the patch solve your problem? [Y/n]
```

If you answer no, ChatCode asks for runtime/user feedback and builds `FOLLOWUP_CONTEXT.md` against the **new** repository state.

You can later regenerate the unresolved follow-up with:

```bash
chatcode followup
```

## `chatcode undo`

```bash
chatcode undo
```

Reverses the latest ChatCode patch using stored history.

The reversal is verified rather than forced. ChatCode does not use destructive recovery commands such as `git reset --hard`.

## `chatcode review`

```bash
chatcode review
```

Opens the latest stored before/after snapshots in VS Code.

To review an older entry:

```bash
chatcode review 2
```

Because history stores snapshots, review does not depend on what the working tree looks like later.

---

# How retrieval works

ChatCode uses multiple retrieval layers rather than trusting one ranking method.

## 1. Persistent static project index

Indexable source types are currently:

```text
.py
.js
.jsx
.ts
.tsx
.java
.kt
.php
```

Each indexed file receives a SHA-256 source hash and compact structural metadata.

Python uses the standard Python AST and records high-level definitions such as:

- classes,
- functions,
- methods,
- command handlers,
- qualified symbol names,
- imports.

Other supported source types use a conservative generic structural parser for top-level symbols and JavaScript/TypeScript-style imports.

Direct file dependencies are resolved from imports where possible.

The index is stored as:

```text
project-map.json
```

inside ChatCode's per-repository workspace, not inside the target repository.

Unchanged files are reused by hash instead of being re-analyzed every run.

## 2. Optional local Qwen semantic metadata

In `ai` mode, eligible source files are also analyzed locally through Ollama.

Qwen is **not** asked to rewrite the repository or invent a dependency graph.

For each eligible file it returns only compact metadata:

```json
{
  "summary": "...",
  "tags": ["..."],
  "important_symbols": ["..."]
}
```

The allowed `important_symbols` are restricted back to symbols ChatCode already found statically.

Semantic results are cached by:

- source hash,
- model name,
- analyzer version.

The deterministic project map is saved before slow model calls begin, and completed semantic files are checkpointed as they finish.

If indexing is interrupted, a later run can resume pending semantic work.

If Ollama/model preflight fails, ChatCode falls back to static retrieval for that run.

## 3. Task-level retrieval

When you run:

```bash
chatcode context "some task"
```

ChatCode combines several signals.

### Explicit targets

Code-shaped references in the task are resolved first when possible, such as:

```text
DashboardAPI
some_function()
/discord-command
```

### Task surfaces

Broader tasks are split into requirement-like surfaces so a multi-layer request does not collapse into one high-scoring file.

This matters for tasks that span areas such as:

```text
backend + API + frontend + tests
```

### Indexed ranking

Paths, symbols, imports and optional Qwen semantic metadata are scored against the task.

Initial matches can pull in shallow indexed dependencies.

### Qwen task hints

In AI mode, Qwen gets a bounded vocabulary of **real indexed symbol IDs** and maps the natural-language task to likely existing symbols.

Qwen is not trusted to invent arbitrary repository paths here.

Returned hints are accepted only when they map back to supplied candidates.

### Structural expansion

ChatCode then expands the selected roots with deterministic relationships such as:

- transport/dispatcher roots,
- implementation calls,
- direct call-site/test relationships,
- selected-file dependencies.

A bounded completeness pass can ask Qwen whether an already-known candidate appears to be missing.

## 4. Source coverage

Finding the correct file is not enough.

A large file may contain many unrelated definitions, while the actual patch requires only a few specific handlers/helpers.

For broader tasks, `source_coverage` ranks concrete definitions inside the files retrieval already selected.

This stage does not perform open-ended project discovery. It asks:

> Inside the files we already trust as relevant, which definitions are needed to cover the requested layers?

## 5. Context contract

The context contract is stricter than ordinary retrieval.

It separates context into concepts such as:

- mandatory patch source,
- high-priority supporting source,
- ordinary support.

For complex tasks it tries to identify complete structural owners rather than isolated keyword matches.

Examples include:

- backend transport/dispatch handlers,
- related serializers/helpers,
- major UI owners,
- representative API tests,
- the correct test fixture/setup.

Python methods can be materialized with qualified locators such as:

```text
DashboardAPI._handle
DashboardAPITests.setUp
```

This prevents an unrelated method with the same short name from being selected accidentally.

If required source cannot be materialized safely, the generated context is marked incomplete instead of silently pretending enough code was provided.

## 6. Exact current-source materialization

The final code sent to ChatGPT is re-read from the current working tree.

Depending on file size and the contract, ChatCode emits source as:

```text
FULL FILE
SYMBOL CONTEXT
EXCERPT
```

Source sections include current line ranges and hashes where applicable.

This means the final patch context is based on what is actually on disk now, including relevant uncommitted work, rather than reconstructing an old version from the index.

---

# Apply and validation model

The model generates a patch, but ChatCode remains the local execution layer.

## Before apply

The candidate diff is parsed and canonicalized.

ChatCode validates:

- repository-relative paths,
- no `..` traversal,
- no absolute paths,
- no `.git` modification,
- sufficient hunk context,
- current-context freshness,
- Git applicability.

No repository mutation occurs if these checks fail.

## Baseline

Before mutation, ChatCode captures the current repository state and reuses a verified pre-patch full test baseline when one is available. A CLI apply does not block to create a missing baseline; its post-patch background suite establishes the verified state for the next apply.

When a baseline is available, a red test after the patch is not automatically a regression if it was already failing before the patch. Without one, ChatCode reports that regression classification is unavailable rather than delaying the apply.

## After apply

ChatCode runs the configured `fast` test profile first. Without one, it tries a narrow relevant Python test selection when it can map changed source to tests safely.

When changed Python production files have no unambiguous conventional test-file match, ChatCode prints a non-blocking warning. This is a discovery warning, not proof that coverage is missing; indirect, parametrized, or differently named tests may still cover the change.

The configured or detected `full` suite then runs in the background and is reused as a verified baseline on the next apply when the repository state is unchanged.

Run `chatcode status` to see whether the latest background full suite is running, passed, failed, or ended with an infrastructure error. Completed status includes its command, duration, report path, and known failing test IDs.

Post-patch failures are classified relative to the baseline as:

- new regressions,
- pre-existing failures,
- unresolved repair targets,
- unclear/infrastructure failures.

The resulting patch and before/after snapshots are stored in ChatCode history.

---

# Test detection

For predictable project-specific behavior, create `.chatcode/tests.toml`:

```toml
[tests]
fast = ["python", "-m", "pytest", "-n", "auto", "-m", "not integration and not e2e"]
full = ["python", "-m", "pytest", "-n", "auto"]
```

Argument lists are recommended because they work without shell parsing. Tokenized command strings are also accepted, but shell operators such as `&&` are not evaluated; use a project script when shell behavior is needed. `fast` is the immediate post-change check; `full` is used for complete validation and baselines. Either profile may be omitted.

When a profile is not configured, ChatCode falls back to automatic detection.

ChatCode currently detects common test setups for:

- npm / pnpm / Yarn / Bun projects
- Maven
- Gradle
- Python (`pytest` or `unittest`)
- Composer

If no supported test command can be identified, ChatCode reports that automated tests are unavailable instead of inventing a command.

Your project's normal dependencies/toolchain still need to be installed locally.

---

# Indexing modes

## AI mode

```env
CHATCODE_INDEX_MODE=ai
CHATCODE_QWEN_MODEL=qwen2.5-coder:1.5b
```

Uses deterministic structure plus local Qwen metadata and task-level semantic hints.

Best choice for normal use.

## Static mode

```env
CHATCODE_INDEX_MODE=static
```

Never invokes Ollama.

Uses paths, symbols, imports, deterministic content/structure and dependency expansion.

Useful when:

- Ollama is unavailable,
- you want the fastest deterministic run,
- you are debugging retrieval without semantic assistance.

## Rebuilding the index

Normally the index updates incrementally.

To force a clean rebuild:

```bash
chatcode status --reindex
```

This is useful after major indexing changes or when debugging an old/stale map.

---

# First run can take longer

The first AI-mode run against a larger repository can take significantly longer because Qwen may need to semantically analyze many eligible files.

Later runs are much faster when files have not changed because semantic results are reused from cache.

Progress is checkpointed file by file.

If you interrupt semantic indexing, rerun the same context command and ChatCode can continue pending work instead of starting the completed semantic work from zero.

---

# Workspace layout

ChatCode keeps generated state outside the target repository in its own workspace.

A repository workspace looks roughly like:

```text
workspace/
└── my-project/
    └── _<repo-hash>/
        ├── UPLOAD_TO_CHATGPT.md
        ├── project-map.json
        ├── patches/
        │   └── incoming.diff
        ├── test-results/
        │   ├── latest.md
        │   ├── baseline.md
        │   ├── targeted.md
        │   └── full-suite.md
        ├── history/
        │   ├── applied/
        │   └── undone/
        ├── PATCH_REPAIR_CONTEXT.md
        ├── CHECK_REPAIR_CONTEXT.md
        ├── FOLLOWUP_CONTEXT.md
        └── followup-state.json
```

Not every file exists at all times.

The workspace path contains a hash of the absolute repository path, so repositories with the same directory name can still receive separate ChatCode state.

---

# Safety and trust boundaries

ChatCode is designed so that ChatGPT produces a patch while local code performs the actual repository operations.

Current safeguards include:

- ignored secret files/directories,
- best-effort redaction of common secret/token/password patterns,
- repository-relative patch path validation,
- `.git` protection,
- hunk/context validation,
- stale-context detection,
- `git apply --check`,
- explicit interactive review/confirmation,
- pre-patch test baseline,
- post-patch targeted/full tests,
- reversible patch history.

Secret detection is **best effort**, not a proof that an export contains no sensitive information.

Review `UPLOAD_TO_CHATGPT.md` before uploading it when working with sensitive repositories.

Likewise, patch applicability and green tests establish technical correctness better than authority. The current implementation does not yet enforce a strict capability policy that limits every patch hunk to only the exact paths/source regions exported to the model.

Treat repository content and model output as untrusted inputs and review important changes before keeping them.

---

# Troubleshooting

## Qwen/Ollama is unavailable

Check:

```bash
ollama --version
ollama list
```

Make sure the configured model exists:

```bash
ollama pull qwen2.5-coder:1.5b
```

If AI preflight fails, ChatCode should continue with static retrieval for that run.

## Initial indexing is taking a long time

This is expected on the first AI-mode pass over a larger repository.

Completed semantic analysis is cached and checkpointed, so later runs should reuse unchanged files.

For debugging or a fast deterministic run:

```env
CHATCODE_INDEX_MODE=static
```

## Retrieval looks stale or strange

Force a clean project-map rebuild:

```bash
chatcode status --reindex
```

Then regenerate the context.

## Patch does not apply

Run:

```bash
chatcode apply
```

A syntax/applicability/stale-context failure creates a repair context with current working-tree information rather than partially applying the patch.

Upload the generated repair context to ChatGPT and use the instructions ChatCode prints.

## Tests fail after apply

ChatCode compares the post-patch failures against the pre-patch baseline.

For a new regression it creates `PATCH_REPAIR_CONTEXT.md`.

Use:

```bash
chatcode repair
```

to expose the latest valid repair context again.

## Tests pass but the feature is still wrong

Answer `n` when ChatCode asks:

```text
Did the patch solve your problem? [Y/n]
```

Describe what is still wrong.

ChatCode creates `FOLLOWUP_CONTEXT.md` against the current source, including the fact that the previous patch applied and validation passed.

---

# Design principles

ChatCode intentionally keeps the responsibilities separate:

```text
Qwen:
semantic hints and compact retrieval metadata

ChatGPT:
reason about supplied code and produce a unified diff

ChatCode:
index, retrieve, materialize current source, validate, apply, test,
review, track history, repair and follow up

Git:
final patch applicability and repository state
```

The goal is not to make a local model write the whole patch.

The goal is to make sure the stronger patch-generating model receives the **smallest useful context that is still complete enough to make a correct change**.

That is the core idea behind Poor Man's Codex.

---

## License

Copyright © 2026 Elliott Olsson. All rights reserved.

No license is granted for commercial use, redistribution, or derivative works without prior written permission.
