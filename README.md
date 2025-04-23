# QA Platform

A lean, local-first web application that enables analysts to QA components of AgentSmyth's financial-analysis platform via OpenAI prompts, record LLM-generated evaluations, and log human pass/fail votes for future insight + RLHF.

## Features (MVP)

- Single-user, local execution (no auth, no hosting)
- Support for all existing agents & sub-components
- Variable-aware prompt execution
- Two-level dropdown UI with report input
- Markdown result pane with pass/fail voting
- Local MongoDB persistence

## Tech Stack

- Frontend: React + Vite + TypeScript
- UI Kit: shadcn/ui + Tailwind
- Backend: FastAPI + uvicorn
- Database: MongoDB (local) using motor
- LLM: OpenAI (gpt-4o-search)

## Getting Started

Coming soon...

## License

Proprietary - All rights reserved 
