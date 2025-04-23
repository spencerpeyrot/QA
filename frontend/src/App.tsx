import { useState } from 'react'
import { ReportVariables } from './components/ReportVariables'
import { QAResponse } from './components/QAResponse'

const AGENT_SUBCOMPONENTS = {
  'S': [], // Agent S has no sub-components
  'M': [
    'Long Term View',
    'Short Term View',
    'Sector Level View',
    'Key Gamma Levels'
  ],
  'Q': [
    'Support & Resistance',
    'Volume Analysis',
    'Candlestick Patterns',
    'Consolidated Analysis'
  ],
  'O': [
    'Flow Analysis',
    'Volatility Analysis',
    'Consolidated Analysis'
  ],
  'E': [
    'EPS/Revenue Analysis',
    'Forward Guidance',
    'Quantitative Analysis',
    'Options Analysis',
    'Consolidated Analysis'
  ],
  'TD': [
    'Overview',
    'Agent S Snapshot',
    'Agent Q Snapshot',
    'Agent O Snapshot',
    'Agent M Snapshot'
  ]
} as const;

// Test data type
interface QARun {
  id: string;
  agent: string;
  sub_component: string | null;
  report_text: string;
  variables: Record<string, string>;
  response_markdown: string;
  qa_rating: number;
  report_rating: number;
  created_at: string;
}

// Test data for development
const TEST_QA_RUN: QARun = {
  id: "test-123",
  agent: "M",
  sub_component: "Long Term View",
  report_text: "Sample report text for testing",
  variables: {
    current_date: "2024-04-22",
    category: "Technology",
    report_text: "Sample report text for testing"
  },
  response_markdown: `
## QA Evaluation Results

### Content Analysis
✅ The report provides a comprehensive long-term market view for the technology sector.
✅ Key trends and market drivers are clearly identified.
⚠️ Consider adding more quantitative data to support the conclusions.

### Technical Assessment
- Market trends analysis is thorough
- Sector comparison is well-structured
- Risk factors are properly identified

### Recommendations
1. Add more specific market data points
2. Include competitor analysis
3. Expand on regulatory impacts

### Overall Rating
**Quality Score**: 8/10
The report meets most quality criteria but could benefit from additional quantitative support.
`,
  qa_rating: 0,
  report_rating: 0,
  created_at: new Date().toISOString()
};

function App() {
  const [selectedAgent, setSelectedAgent] = useState<string>('');
  const [selectedSubComponent, setSelectedSubComponent] = useState<string>('');
  const [variables, setVariables] = useState<Record<string, string>>({});
  const [qaRun, setQaRun] = useState<QARun | null>(null);

  const handleAgentChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const agent = e.target.value;
    setSelectedAgent(agent);
    setSelectedSubComponent(''); // Reset sub-component when agent changes
  };

  const handleVariablesChange = (newVariables: Record<string, string>) => {
    setVariables(newVariables);
  };

  const handleQARating = (id: string, rating: number) => {
    if (qaRun && qaRun.id === id) {
      setQaRun({ ...qaRun, qa_rating: rating });
    }
  };

  const handleReportRating = (id: string, rating: number) => {
    if (qaRun && qaRun.id === id) {
      setQaRun({ ...qaRun, report_rating: rating });
    }
  };

  const loadTestData = () => {
    setQaRun(TEST_QA_RUN);
  };

  const getAgentColor = (agent: string) => {
    switch(agent) {
      case 'M': return 'var(--color-agentM-light)';
      case 'S': return 'var(--color-agentS-light)';
      case 'O': return 'var(--color-agentO-light)';
      case 'E': return 'var(--color-agentE-light)';
      case 'Q': return 'var(--color-agentQ-light)';
      default: return 'var(--color-accent)';
    }
  };

  return (
    <div className="min-h-screen bg-(--color-background)">
      <div className="p-6">
        <header className="mb-8">
          <h1 className="font-owners text-3xl font-semibold text-(--color-foreground)">
            QA Platform
          </h1>
        </header>
        
        <main className="space-y-6">
          <div 
            className="space-y-6 rounded-lg bg-(--color-background) p-6 shadow-lg relative overflow-hidden border border-[#2A2E39]"
            style={{
              backgroundImage: 'radial-gradient(circle, #c1ff0005 1px, transparent 1px)',
              backgroundSize: '4px 4px'
            }}
          >
            <div className="space-y-4 relative z-10">
              <div className="flex items-center gap-4">
                <label 
                  htmlFor="agent" 
                  className="w-24 font-medium text-(--color-neutral-100)"
                >
                  Agent:
                </label>
                <select 
                  id="agent" 
                  value={selectedAgent} 
                  onChange={handleAgentChange}
                  className="w-full rounded-md border border-[#2A2E39] bg-[#1a1d24] px-3 py-2 text-(--color-neutral-100) focus:outline-none focus:border-(--color-accent)"
                >
                  <option value="">Select an agent...</option>
                  <option value="S">Agent S</option>
                  <option value="M">Agent M</option>
                  <option value="Q">Agent Q</option>
                  <option value="O">Agent O</option>
                  <option value="E">Agent E</option>
                  <option value="TD">Ticker Dashboard</option>
                </select>
              </div>

              {selectedAgent && (
                <div 
                  className="rounded-md px-4 py-3"
                  style={{ backgroundColor: `${getAgentColor(selectedAgent)}20` }}
                >
                  <p 
                    className="text-sm font-medium"
                    style={{ color: getAgentColor(selectedAgent) }}
                  >
                    Selected: Agent {selectedAgent}
                  </p>
                </div>
              )}
            </div>

            {selectedAgent && AGENT_SUBCOMPONENTS[selectedAgent as keyof typeof AGENT_SUBCOMPONENTS].length > 0 && (
              <div className="flex items-center gap-4">
                <label 
                  htmlFor="subcomponent"
                  className="w-24 font-medium text-(--color-neutral-100)"
                >
                  Sub-component:
                </label>
                <select
                  id="subcomponent"
                  value={selectedSubComponent}
                  onChange={(e) => setSelectedSubComponent(e.target.value)}
                  className="w-full rounded-md border border-[#2A2E39] bg-[#1a1d24] px-3 py-2 text-(--color-neutral-100) focus:outline-none focus:border-(--color-accent)"
                >
                  <option value="">Select a sub-component...</option>
                  {AGENT_SUBCOMPONENTS[selectedAgent as keyof typeof AGENT_SUBCOMPONENTS].map((subComponent) => (
                    <option key={subComponent} value={subComponent}>
                      {subComponent}
                    </option>
                  ))}
                </select>
              </div>
            )}

            {selectedAgent && selectedSubComponent && (
              <div 
                className="rounded-md px-4 py-3"
                style={{ backgroundColor: `${getAgentColor(selectedAgent)}20` }}
              >
                <p 
                  className="text-sm font-medium"
                  style={{ color: getAgentColor(selectedAgent) }}
                >
                  Selected: Agent {selectedAgent} → {selectedSubComponent}
                </p>
              </div>
            )}
          </div>

          {/* Report Variables Section */}
          {selectedAgent && (
            <div 
              className="rounded-lg bg-(--color-background) p-6 shadow-lg relative overflow-hidden border border-[#2A2E39]"
              style={{
                backgroundImage: 'radial-gradient(circle, #c1ff0005 1px, transparent 1px)',
                backgroundSize: '4px 4px'
              }}
            >
              <div className="relative z-10">
                <ReportVariables
                  agent={selectedAgent}
                  subComponent={selectedSubComponent}
                  onVariablesChange={handleVariablesChange}
                />
              </div>
            </div>
          )}

          {/* Action Buttons */}
          {selectedAgent && Object.keys(variables).length > 0 && (
            <div className="flex justify-end gap-4">
              <button
                className="rounded-md bg-[#1a1d24] px-6 py-2 text-sm font-medium text-(--color-neutral-100) border border-[#2A2E39] hover:border-(--color-accent)"
                onClick={loadTestData}
              >
                Load Test Data
              </button>
              <button
                className="rounded-md bg-(--color-accent) px-6 py-2 text-sm font-medium text-(--color-neutral-900) hover:bg-opacity-90 focus:outline-none focus:ring-2 focus:ring-(--color-accent)"
                onClick={() => console.log('Variables:', variables)}
              >
                Run QA
              </button>
            </div>
          )}

          {/* QA Response Section */}
          {qaRun && (
            <QAResponse
              qaRun={qaRun}
              onQARating={handleQARating}
              onReportRating={handleReportRating}
            />
          )}
        </main>
      </div>
    </div>
  )
}

export default App
