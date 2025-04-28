# QA Platform

A lean, local-first web application that enables analysts to QA components of AgentSmyth's financial-analysis platform via OpenAI prompts, record LLM-generated evaluations, and log human quality ratings for future insight + RLHF.

## Features

- **Modern Dark Theme UI**: Sleek, professional dark mode interface optimized for long analysis sessions
- **Interactive Rating System**: 5-star rating system for both QA quality and report accuracy
- **Agent & Component Selection**: Two-level dropdown for precise agent and sub-component targeting
- **Variable Management**: Dynamic form generation based on component requirements 
- **Real-time LLM Integration**: Streaming responses from OpenAI's GPT-4-Search 
- **Markdown Rendering**: Beautiful display of LLM-generated analysis 
- **Persistent Storage**: MongoDB integration for historical analysis 

## Tech Stack

### Frontend
- **Framework**: React 18 + Vite + TypeScript
- **Styling**: Tailwind CSS with custom dark theme
- **UI Components**: Custom components with Lucide icons
- **State Management**: React Context API
- **Build Tools**: Vite, ESLint, TypeScript

### Backend
- **API**: FastAPI with async support
- **Database**: MongoDB with Motor (async driver)
- **LLM Integration**: OpenAI API (gpt-4-search)
- **Runtime**: Python 3.12 + uvicorn

## Project Structure

```
QA/
├── frontend/           # React + Vite frontend
│   ├── src/
│   │   ├── components/  # Reusable UI components
│   │   ├── App.tsx     # Main application component
│   │   └── ...
│   └── ...
├── backend/           # FastAPI backend
│   ├── api/          # API routes
│   ├── models/       # Data models
│   └── ...
└── prompts/          # QA prompt templates
```

## Getting Started

### Prerequisites
- Node.js 18+
- Python 3.12+
- MongoDB
- OpenAI API key

### Development Setup
1. Clone the repository
2. Set up the frontend:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
3. Set up the backend:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   uvicorn main:app --reload
   ```
4. Configure environment variables:
   - Create `.env` in the backend directory
   - Add your OpenAI API key: `OPENAI_API_KEY=your_key_here`

## License

Proprietary - All rights reserved 
