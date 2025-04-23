# QA Platform Frontend

The frontend implementation of the QA Platform, built with React, Vite, and TypeScript. Features a modern dark theme and interactive components for QA analysis.

## Features

- **Dark Theme**: Professional dark mode interface with custom color variables
- **Interactive Components**:
  - Star Rating System (0-5 stars)
  - Agent & Sub-component Selection
  - Dynamic Variable Forms
  - Markdown Response Display
- **Responsive Design**: Optimized for desktop workstations

## Tech Stack

- **Core**: React 18 + TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS
- **Icons**: Lucide React
- **Code Quality**: ESLint + TypeScript

## Project Structure

```
src/
├── components/           # Reusable UI components
│   ├── AgentSelector    # Agent selection dropdown
│   ├── QAResponse      # QA result display with ratings
│   ├── ReportVariables # Dynamic variable form
│   └── StarRating      # Interactive star rating
├── App.tsx             # Main application component
├── index.css          # Global styles and Tailwind
└── main.tsx           # Application entry point
```

## Component Documentation

### QAResponse
Displays the QA evaluation result with:
- Metadata header (timestamp, ID)
- Agent and sub-component info
- Markdown-rendered LLM response {COMING SOON}
- Dual star rating system for QA quality and report accuracy

### StarRating
Interactive 5-star rating component with:
- Hover effects
- Click-to-rate functionality
- Visual feedback
- Accessibility features

### ReportVariables
Dynamic form generation based on component requirements:
- Validates required fields
- Supports multiple input types
- Real-time validation

### AgentSelector
Two-level dropdown for agent and sub-component selection:
- Contextual sub-component options
- Clear visual hierarchy
- Keyboard navigation support

## Development

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start development server:
   ```bash
   npm run dev
   ```

3. Build for production:
   ```bash
   npm run build
   ```

## Style Guide

- Follow the color system defined in `globals.css`
- Use Tailwind classes for styling
- Maintain dark theme consistency
- Follow existing component patterns

## Contributing

1. Follow the established code style
2. Maintain TypeScript type safety
3. Test components before committing
4. Update documentation as needed

## License

Proprietary - All rights reserved
