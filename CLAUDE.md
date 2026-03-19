# NM i AI 2026 - Competition Repository

## Context
This is our entry for NM i AI 2026 (Norwegian AI Championship).
- **Dates:** March 19 18:00 CET - March 22 15:00 CET (69 hours)
- **Prize pool:** 1,000,000 NOK
- **Platform:** https://app.ainm.no

## Directive
Every submission is ONE SHOT. There is no room for error.
- Ground every assumption in reproducible testing
- Use internal rubrics to validate before submitting
- Confidence cannot be compromised - verify, then verify again
- Excellence is the only acceptable standard

## Three Tasks (33% each)

| Task | Type | Submission |
|------|------|------------|
| Tripletex | AI accounting agent | HTTPS endpoint `/solve` |
| Astar Island | Norse world prediction | REST API predictions |
| NorgesGruppen Data | Object detection | ZIP code upload |

## Documentation
All competition docs are in `docs/` - see `docs/README.md` for index.

## MCP Server
```bash
claude mcp add --transport http nmiai https://mcp-docs.ainm.no/mcp
```

## Key Constraints
- Code must be open-sourced (MIT) in public repo before deadline
- Vipps verification required for prizes
- Rate limits vary by task - check docs
- Deploy in `europe-north1` for lowest latency
