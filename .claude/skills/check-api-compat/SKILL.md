---
name: check-api-compat
description: >
  Use this skill when the user modifies protocol files in the OpenAI or Anthropic
  entrypoints, asks to "check API compatibility", "verify spec compliance", or
  discusses API parameter changes. Fetches upstream specs and compares parameters.
---

# Check API Compatibility

Verify that API parameter changes in vLLM's OpenAI/Anthropic-compatible entrypoints comply with the upstream spec.

## Steps

1. **Find modified protocol files.** Run `git diff --name-only HEAD` (or use a user-specified file) and filter for `protocol.py` files under `vllm/entrypoints/openai/` or `vllm/entrypoints/anthropic/`.

2. **Extract changed parameters.** For each modified protocol file, run `git diff HEAD -- <file>` and identify added, removed, or modified Pydantic fields in request/response models.

3. **Find the spec URL.** Search the protocol file for a comment containing `platform.openai.com` (OpenAI) or `docs.anthropic.com` (Anthropic). This is the upstream spec reference.

4. **Fetch the upstream spec.** Use web search or web fetch with the spec URL to retrieve the current parameter definitions for the relevant endpoint.

5. **Compare each parameter** against the upstream spec:
   - **Name**: Must match exactly.
   - **Type**: Must match (or be a compatible superset if intentionally extending).
   - **Default value**: Must match the spec default.
   - **Required/optional**: Must match.
   - **Semantics**: Field description should align with spec behavior.

6. **Classify and report results:**
   - **Matches spec**: Parameter name, type, and default all align.
   - **Diverges from spec**: Show what the spec says vs. what the code has.
   - **vLLM extension**: Fields under a `# vLLM-specific fields` comment are exempt.
   - **Not in spec**: Parameter exists in code but not found in upstream spec (flag for review).

7. **If web search is unavailable**, output the spec URLs found in the protocol files and note: "API parameter changes not verified against live spec -- please verify manually."

## Output

After completing the steps above, show the user a report with:

```markdown
## API Compatibility Report

**Result: PASS / FAIL**

File: vllm/entrypoints/openai/chat_completion/protocol.py
Spec: https://platform.openai.com/docs/api-reference/chat/create

| Parameter          | Status           | Notes                                |
|--------------------|------------------|--------------------------------------|
| temperature        | Matches spec     | Optional[float], default=None        |
| my_new_param       | NOT IN SPEC      | Not found in OpenAI spec             |
| kv_transfer_params | vLLM extension   | Exempt (labeled as vLLM-specific)    |

### Conclusion
<1-2 sentences: whether all changes comply, and what needs fixing if not>

### References
- OpenAI API: https://platform.openai.com/docs/api-reference
- Anthropic API: https://docs.anthropic.com/en/api
- Protocol file spec URLs found in source comments
```

- **PASS**: All changed parameters match the upstream spec (or are labeled vLLM extensions).
- **FAIL**: Any parameter diverges from spec or is missing from spec without a vLLM-extension label. List each failing parameter with what the spec expects.
