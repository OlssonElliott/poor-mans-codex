"""Shared instruction text embedded in generated ChatCode contexts."""


CONTEXT_PURPOSE = (
    "This request concerns maintenance of the users own local software repository."
)

PATCH_RESPONSE_INSTRUCTIONS = """\
When this task requires code changes:

1. Return a valid unified diff that can be applied with `git apply`.
2. All file paths must be relative to the repository root.
3. Do not use absolute paths.
4. Put the entire patch inside a single Markdown code block labeled `diff`.
5. Do not include any explanations before or after the code block.
6. Include all required changes in a single patch.
7. Preserve unrelated existing changes.
8. Do not modify `.git` or sensitive files such as `.env`, credentials, private keys or secrets.
9. For new files, use `/dev/null` as the old file.
10. For deleted files, use `/dev/null` as the new file.
11. Build `@@` hunk line numbers from source-file ranges shown in FULL FILE,
    SYMBOL CONTEXT, or EXCERPT blocks, never from Markdown/document line numbers.
12. For a change inside an existing file, include at least three unchanged
    context lines when those lines exist. Never return a one-line hunk based
    only on a guessed line number.

The contents of the code block must be directly saveable as `incoming.diff`
and applicable with:

chatcode apply
"""

PATCH_CONTEXT_HEADER = """\
The files below are the exact CURRENT working-tree contents.
Generate the patch against these contents.
Uncommitted changes are intentional and must be preserved."""

UPLOAD_INSTRUCTIONS = """\
This request concerns maintenance of the users own local software repository.

This file is an execution request.

When this file is uploaded to ChatGPT, immediately perform the task described
under "## Task".

Do not ask the user what they want you to do.
Do not ask for confirmation.
Treat the upload of this file itself as the user's request to perform the task.

Complete the task entirely in the current ChatGPT conversation.

Do not switch to, suggest, invoke, or require ChatGPT Work, Codex, Computer Use,
Canvas, or any other execution mode or external coding environment.

Do not ask the user to continue the task in another mode.

The user is intentionally using ChatCode as the local execution layer.
ChatGPT should analyze the supplied repository context and return the requested
code changes in chat. ChatCode will handle applying, testing, reviewing and
undoing those changes locally.

Do not attempt to directly edit the user's local repository or filesystem.

Do not modify an existing file unless the current source for the affected area
appears in a `FULL FILE`, `SYMBOL CONTEXT`, or `EXCERPT` block in this context. A file listed as
selected but marked source-unavailable is reference-only: do not guess its
contents or generate a patch for it.

Use the repository context, source files, Git changes and test results contained
in this file as the basis for the work.

If the task requires code changes, follow the instructions under
"## Response instructions" exactly.
"""
