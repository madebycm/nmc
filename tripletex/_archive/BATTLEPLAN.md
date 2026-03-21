# Tripletex AI Agent — BATTLEPLAN

## Architecture

```
Task Prompt
    ↓
Gemini 2.5 Pro (35 typed tools generated from OpenAPI spec)
    ↓
Tool Router (maps tool calls → HTTP requests)
    ↓
Schema Guard (final validation against spec)
    ↓
Tripletex REST API
    ↓
JSONL task log (every call recorded)
```

**Key insight**: One typed tool per API operation. The LLM CANNOT hallucinate field names — tool parameters ARE the schema. Generated automatically from the real OpenAPI spec (379 endpoints, 1035 schemas).

35 tools, 38KB. Gemini supports 512. No dynamic selection needed.

## Endpoint: `https://nm.j6x.com/solve`

## Deploy
```bash
scp ~/www/nm/tripletex/{agent.py,prompts.py,schema_guard.py,tool_router.py,api_spec_extract.json,gen_tools_output.json} vps:/opt/tripletex/
ssh vps 'systemctl restart tripletex'
```

## Scoring
- 30 task types × tier multiplier (T1=×1, T2=×2, T3=×3)
- Efficiency bonus up to 2× (perfect correctness only)
- Best score per task kept forever
- 10 attempts/task/day, resets midnight UTC
- 3 concurrent submissions

## Files
| File | Purpose |
|------|---------|
| agent.py | Gemini agent loop, typed tool dispatch |
| tool_router.py | Maps typed tool calls → HTTP method/endpoint/body |
| schema_guard.py | Final validation layer against OpenAPI spec |
| prompts.py | Slim system prompt (rules only, no API reference) |
| gen_tools.py | Generates typed tools from OpenAPI spec |
| gen_tools_output.json | 35 typed tool declarations |
| api_spec_extract.json | Full OpenAPI spec extract |
| server.py | FastAPI /solve endpoint |
| validator.py | Local self-validator against sandbox |
