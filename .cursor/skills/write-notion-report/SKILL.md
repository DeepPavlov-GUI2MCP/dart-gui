---
name: write-notion-report
description: Create structured report pages in Notion and include images when needed. Use when the user asks to write a report, failure analysis, or investigation summary into Notion. If the report needs inline images, upload them through the Notion file upload API and append image blocks; otherwise use the Notion MCP tools only.
disable-model-invocation: true
---

# Write Notion Report

Use this skill when creating new report-style documents in Notion from this workspace.

## Default behavior

- If the report is text-only, use the Notion MCP tools only.
- If the report must contain inline images, use the Notion MCP tools for page creation/content and the direct Notion API for file uploads + image block insertion.

## Before using Notion MCP

1. Read the Notion MCP tool descriptors before calling them.
2. Fetch the Notion enhanced markdown spec resource before writing page content.
3. Search for the target parent page or teamspace first.
4. If a parent page name is ambiguous, ask the user to clarify before creating the page.

## Text-only report workflow

1. Find the target parent in Notion.
2. Create the page with `notion-create-pages`.
3. Add or refine content with `notion-update-page` if needed.
4. Return the page URL and the parent location.

## Report-with-images workflow

Use this only if the report needs inline screenshots or other inline media.

### Authentication

- Assume `NOTION_API_KEY` may already be available in the shell environment.
- Check whether `NOTION_API_KEY` is set before using the direct API.
- If it is missing and images are required, ask the user to provide one.
- If the user wants persistence, add it to `~/.bashrc`; otherwise prefer a temporary shell export for the current session.
- Do not store the token in repository files.

### API version

Use:

```bash
Notion-Version: 2026-03-11
```

### Page creation

1. Create the page body with Notion MCP first.
2. Keep the report content complete even before images are attached.
3. Add explicit image section headings so later image blocks have a clear place in the document.

### Image upload flow

For each image file:

1. Create a file upload object:

```bash
curl -sS -X POST 'https://api.notion.com/v1/file_uploads' \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H 'Notion-Version: 2026-03-11' \
  -H 'Content-Type: application/json' \
  --data '{"mode":"single_part","filename":"example.png","content_type":"image/png"}'
```

2. Extract the returned `id`.
3. Send the file contents:

```bash
curl -sS -X POST "https://api.notion.com/v1/file_uploads/$FILE_UPLOAD_ID/send" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H 'Notion-Version: 2026-03-11' \
  -F "file=@/absolute/path/to/example.png;type=image/png"
```

4. Verify the returned file upload object has `status` equal to `uploaded`.

### Insert uploaded images into the page

Append image blocks to the report page with the uploaded file IDs:

```bash
curl -sS -X PATCH "https://api.notion.com/v1/blocks/$PAGE_ID/children" \
  -H "Authorization: Bearer $NOTION_API_KEY" \
  -H 'Notion-Version: 2026-03-11' \
  -H 'Content-Type: application/json' \
  --data '{
    "children": [
      {
        "object": "block",
        "type": "image",
        "image": {
          "type": "file_upload",
          "file_upload": { "id": "FILE_UPLOAD_ID" },
          "caption": [
            {
              "type": "text",
              "text": { "content": "Example caption" }
            }
          ]
        }
      }
    ]
  }'
```

### Verification

After appending image blocks:

1. Fetch the page again.
2. Confirm the image blocks appear in the fetched content.
3. Return the final page URL.
4. Mention whether the images were added through MCP or direct API.

## Preferred shell behavior

- Prefer a temporary shell export for `NOTION_API_KEY` unless the user explicitly asks to persist it.
- When running Python or scripts in the terminal, activate `.venv` first if needed by the command.

## Report structure

Use a concise report structure unless the user asks for something else:

```markdown
# Summary

# Original task

# What happened

# Root cause

# Evidence

# Recommended follow-up
```

## Important constraints

- Do not assume local file paths can be embedded directly in Notion page markdown.
- Do not assume data URIs will survive Notion page updates.
- If inline images are required, use the upload API and append image blocks.
- If images are optional, prefer the simpler MCP-only route.
