import { useState, useRef, useEffect } from 'react'
import { ReportVariables } from './components/ReportVariables'
import { QAResponse } from './components/QAResponse'
import { Analytics } from './components/Analytics'
import { AutomatedQA } from './components/AutomatedQA'
import AutomatedAnalytics from './components/AutomatedAnalytics'
import { qaApi } from './services/api'

const AGENT_SUBCOMPONENTS = {
  'Agent S': [], // Agent S has no sub-components
  'Agent M': [
    'Long Term View',
    'Short Term View',
    'Sector Level View',
    'Key Gamma Levels'
  ],
  'Agent Q': [
    'Support & Resistance',
    'Volume Analysis',
    'Candlestick Patterns',
    'Consolidated Analysis'
  ],
  'Agent O': [
    'Flow Analysis',
    'Volatility Analysis',
    'Consolidated Analysis'
  ],
  'Agent E': [
    'EPS/Revenue Analysis',
    'Forward Guidance',
    'Quantitative Analysis',
    'Options Analysis',
    'Consolidated Analysis'
  ],
  'Ticker Dashboard': [
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
  qa_rating: boolean | null;
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
**Section 1: Contextual Relevance & Coherence** **Rating:** Good **Justification:** The response effectively addresses the question by detailing Meta Platforms' reaction to the €200 million fine under the Digital Markets Act (DMA), highlighting legal resistance, strategic adjustments, and implications for compliance costs and user engagement in Europe. The information is logically organized, with clear transitions between topics such as legal pushback, compliance costs, user engagement risks, revenue pressures, and political escalation. However, the response could benefit from a more explicit connection between Meta's actions and the broader context of EU regulatory enforcement. **Flag Status:** Not flagged **Section 2: Factual Accuracy** **Rating:** Good **Justification:** The majority of factual claims are accurate and supported by credible sources. For instance, Meta's Chief Global Affairs Officer, Joel Kaplan, has indeed framed the fine as a "multi-billion-dollar tariff" and accused the EU of discriminatory enforcement favoring European and Chinese competitors. Additionally, Meta introduced a revised ad model in November 2024 that uses less personal data, which is still under review by EU regulators. ([reuters.com](https://www.reuters.com/sustainability/boards-policy-regulation/what-happens-apple-meta-after-eu-fine-2025-04-23/?utm_source=openai)) However, some claims lack direct verification. For example, the assertion that Meta has halved subscription prices to €4.99/month in some cases to address EU concerns is not directly supported by the provided sources. The sources indicate a reduction from €9.99 to €5.99 per month on the web and from €12.99 to €7.99 per month on iOS and Android, but do not confirm a halving to €4.99. ([about.fb.com](https://about.fb.com/news/2024/11/facebook-and-instagram-to-offer-subscription-for-no-ads-in-europe/?utm_source=openai)) **Flag Status:** Not flagged **Section 3: Completeness & Depth** **Rating:** Good **Justification:** The response thoroughly covers the key aspects of the question, including Meta's legal and political pushback, adjustments to its ad model, potential revenue impacts, and the broader political context. It provides sufficient detail and analysis, referencing specific actions taken by Meta and the EU's regulatory stance. However, the response could delve deeper into the potential long-term effects on Meta's market position and user behavior in Europe. **Flag Status:** Not flagged **Section 4: Overall Quality & Presentation** **Rating:** Good **Justification:** The response is well-written and clearly presented, with a logical flow of ideas and minimal errors. It effectively summarizes key points and provides actionable insights into Meta's strategic responses and the implications for its European operations. The inclusion of specific data points, such as subscription price reductions and stock price movements, adds depth to the analysis. However, the response could benefit from a more concise summary to reinforce the main takeaways. **Flag Status:** Not flagged **Overall Summary and Recommendations:** **Overall Evaluation:** The response provides a comprehensive and coherent analysis of Meta Platforms' reaction to the €200 million EU fine under the Digital Markets Act, effectively addressing the question with relevant details and insights. **Specific Improvement Recommendations:** - **Clarify Subscription Pricing Details:** Ensure that all claims regarding subscription price reductions are accurately supported by credible sources. - **Expand on Long-Term Implications:** Provide a more in-depth analysis of the potential long-term effects on Meta's market position and user behavior in Europe. - **Enhance Summary:** Include a concise summary at the end to reinforce the main takeaways and implications of the analysis. **Final Action:** No sections require further human review.
`,
  qa_rating: null,
  report_rating: 0,
  created_at: new Date().toISOString()
};

function App() {
  const [selectedTab, setSelectedTab] = useState<'qa' | 'analytics' | 'automated' | 'automated_analytics'>('qa');
  const [selectedAgent, setSelectedAgent] = useState<string>('');
  const [selectedSubComponent, setSelectedSubComponent] = useState<string>('');
  const [variables, setVariables] = useState<Record<string, string>>({});
  const [qaRun, setQaRun] = useState<QARun | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const qaResponseRef = useRef<HTMLDivElement>(null);
  const [shouldScroll, setShouldScroll] = useState(false);
  const [retryCompleted, setRetryCompleted] = useState(false);

  // Track whether there is an active QA run
  const hasActiveQA = Boolean(qaRun && !isLoading);

  // Modified scroll effect
  useEffect(() => {
    if (qaResponseRef.current) {
      if (shouldScroll) {
        // Smooth scroll for initial QA run
        qaResponseRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
        setShouldScroll(false); // Reset after scrolling
      } else if (retryCompleted) {
        // Instant scroll for retry completion
        qaResponseRef.current.scrollIntoView({ behavior: 'instant', block: 'start' });
        setRetryCompleted(false); // Reset after scrolling
      }
    }
  }, [shouldScroll, retryCompleted, qaRun]); // Add retryCompleted and qaRun to dependencies

  const handleAgentChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const agent = e.target.value;
    setSelectedAgent(agent);
    setSelectedSubComponent(''); // Reset sub-component when agent changes
  };

  const handleVariablesChange = (newVariables: Record<string, string>) => {
    setVariables(newVariables);
  };

  const handleRunQA = async () => {
    try {
      setIsLoading(true);
      setError(null);
      setShouldScroll(true); // Set scroll flag for initial QA run

      // Show loading state immediately
      setQaRun({
        id: 'loading',
        agent: selectedAgent,
        sub_component: selectedSubComponent || null,
        report_text: variables.report_text || '',
        variables: variables,
        response_markdown: 'Processing...',
        qa_rating: null,
        report_rating: 0,
        created_at: new Date().toISOString()
      });

      // Format the request data to match backend expectations
      const requestData = {
        agent: selectedAgent,
        sub_component: selectedSubComponent || '', // Empty string instead of undefined
        variables: {
          ...variables,
          current_date: new Date().toISOString().split('T')[0]  // Always use current date
        }
      };

      console.log('Sending QA request:', requestData);

      const response = await qaApi.createQAEvaluation(requestData);
      console.log('QA creation response:', response);

      if (!response?.id) {
        throw new Error('No evaluation ID received from server');
      }

      // Add a small delay to allow backend processing
      await new Promise(resolve => setTimeout(resolve, 1000));

      try {
        // Fetch the full QA evaluation details
        const qaEvaluation = await qaApi.getQAEvaluation(response.id);
        
        setQaRun({
          id: qaEvaluation._id,
          agent: qaEvaluation.agent,
          sub_component: qaEvaluation.sub_component || null,
          report_text: qaEvaluation.variables.report_text || '',
          variables: qaEvaluation.variables,
          response_markdown: qaEvaluation.response_markdown,
          qa_rating: null,
          report_rating: 0,
          created_at: qaEvaluation.created_at
        });
      } catch (fetchError: any) {
        // If we can't fetch the evaluation, still show what we got from creation
        console.warn('Error fetching QA evaluation:', fetchError);
        setQaRun({
          id: response.id,
          agent: selectedAgent,
          sub_component: selectedSubComponent || null,
          report_text: variables.report_text || '',
          variables: variables,
          response_markdown: response.markdown || 'Processing...',
          qa_rating: null,
          report_rating: 0,
          created_at: new Date().toISOString()
        });
      }
    } catch (err) {
      const axiosError = err as any;
      const errorMessage = axiosError.response?.data?.detail || 
                         axiosError.message || 
                         'An error occurred while running QA';
      setError(errorMessage);
      console.error('QA Error Details:', {
        message: axiosError.message,
        response: axiosError.response?.data,
        status: axiosError.response?.status,
        data: axiosError.response?.config?.data
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleQARating = async (id: string, isGood: boolean) => {
    try {
      await qaApi.updateQAPass(id, isGood);
      if (qaRun && qaRun.id === id) {
        setQaRun({ ...qaRun, qa_rating: isGood });
      }
    } catch (err) {
      console.error('Error updating QA rating:', err);
    }
  };

  const handleReportRating = async (id: string, rating: number) => {
    try {
      await qaApi.updateReportPass(id, rating);
      if (qaRun && qaRun.id === id) {
        setQaRun({ ...qaRun, report_rating: rating });
      }
    } catch (err) {
      console.error('Error updating report rating:', err);
    }
  };

  const handleRetry = async () => {
    if (!qaRun) return;
    
    try {
      setIsLoading(true);
      setError(null);

      // Show loading state immediately while preserving the original qaRun's data
      setQaRun({
        ...qaRun,
        qa_rating: null,
        report_rating: 0,
        response_markdown: 'Processing...',
        created_at: new Date().toISOString()
      });

      // Use the same request data from the previous run
      const requestData = {
        agent: qaRun.agent,
        sub_component: qaRun.sub_component || '',
        variables: qaRun.variables
      };

      const response = await qaApi.createQAEvaluation(requestData);
      
      if (!response?.id) {
        throw new Error('No evaluation ID received from server');
      }

      // Add a small delay to allow backend processing
      await new Promise(resolve => setTimeout(resolve, 1000));

      try {
        // Fetch the full QA evaluation details
        const qaEvaluation = await qaApi.getQAEvaluation(response.id);
        
        setQaRun({
          id: qaEvaluation._id,
          agent: qaEvaluation.agent,
          sub_component: qaEvaluation.sub_component || null,
          report_text: qaEvaluation.variables.report_text || '',
          variables: qaEvaluation.variables,
          response_markdown: qaEvaluation.response_markdown,
          qa_rating: null,
          report_rating: 0,
          created_at: qaEvaluation.created_at
        });

        // Trigger scroll effect after successful retry
        setRetryCompleted(true);

      } catch (fetchError: any) {
        console.warn('Error fetching QA evaluation:', fetchError);
        setQaRun({
          id: response.id,
          agent: qaRun.agent,
          sub_component: qaRun.sub_component,
          report_text: qaRun.variables.report_text || '',
          variables: qaRun.variables,
          response_markdown: response.markdown || 'Processing...',
          qa_rating: null,
          report_rating: 0,
          created_at: new Date().toISOString()
        });

        // Trigger scroll effect even after failed fetch
        setRetryCompleted(true);
      }
    } catch (err) {
      const axiosError = err as any;
      const errorMessage = axiosError.response?.data?.detail || 
                         axiosError.message || 
                         'An error occurred while running QA';
      setError(errorMessage);
      console.error('QA Error Details:', {
        message: axiosError.message,
        response: axiosError.response?.data,
        status: axiosError.response?.status,
        data: axiosError.response?.config?.data
      });
    } finally {
      setIsLoading(false);
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

  const handleSubmit = () => {
    // Reset the form after submission
    setQaRun(null);
    setSelectedAgent('');
    setSelectedSubComponent('');
    setVariables({});
  };

  return (
    <div className="min-h-screen bg-(--color-background)">
      <div className="p-6">
        <header className="mb-8">
          <h1 className="font-owners text-3xl font-semibold text-(--color-foreground) mb-6">
            QA Platform
          </h1>
          <div className="flex gap-4 border-b border-[#2A2E39]">
            <button
              className={`px-4 py-2 font-medium text-sm transition-colors relative ${
                selectedTab === 'qa'
                  ? 'text-(--color-accent)'
                  : 'text-(--color-neutral-500) hover:text-(--color-neutral-100)'
              }`}
              onClick={() => setSelectedTab('qa')}
            >
              QA
              {selectedTab === 'qa' && (
                <div className="absolute bottom-0 left-0 w-full h-0.5 bg-(--color-accent)" />
              )}
            </button>
            <button
              className={`px-4 py-2 font-medium text-sm transition-colors relative ${
                selectedTab === 'analytics'
                  ? 'text-(--color-accent)'
                  : 'text-(--color-neutral-500) hover:text-(--color-neutral-100)'
              }`}
              onClick={() => setSelectedTab('analytics')}
            >
              Analytics
              {selectedTab === 'analytics' && (
                <div className="absolute bottom-0 left-0 w-full h-0.5 bg-(--color-accent)" />
              )}
            </button>
            <button
              className={`px-4 py-2 font-medium text-sm transition-colors relative ${
                selectedTab === 'automated'
                  ? 'text-(--color-accent)'
                  : 'text-(--color-neutral-500) hover:text-(--color-neutral-100)'
              }`}
              onClick={() => setSelectedTab('automated')}
            >
              Automated QA
              {selectedTab === 'automated' && (
                <div className="absolute bottom-0 left-0 w-full h-0.5 bg-(--color-accent)" />
              )}
            </button>
            <button
              className={`px-4 py-2 font-medium text-sm transition-colors relative ${
                selectedTab === 'automated_analytics'
                  ? 'text-(--color-accent)'
                  : 'text-(--color-neutral-500) hover:text-(--color-neutral-100)'
              }`}
              onClick={() => setSelectedTab('automated_analytics')}
            >
              Automated Analytics
              {selectedTab === 'automated_analytics' && (
                <div className="absolute bottom-0 left-0 w-full h-0.5 bg-(--color-accent)" />
              )}
            </button>
          </div>
        </header>

        <main className="space-y-6">
          {selectedTab === 'qa' ? (
            <>
              <div className="space-y-6 rounded-lg bg-(--color-background) p-6 shadow-lg relative overflow-hidden border border-[#2A2E39]"
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
                    <div className="relative flex-1">
                      <select 
                        id="agent" 
                        value={selectedAgent} 
                        onChange={handleAgentChange}
                        className="w-full rounded-md border border-[#2A2E39] bg-(--color-background) px-3 py-2 text-(--color-neutral-100) focus:outline-none focus:border-(--color-accent) appearance-none cursor-pointer"
                      >
                        <option value="" className="bg-(--color-background)">Select an agent...</option>
                        <option value="Agent S" className="bg-(--color-background)">Agent S</option>
                        <option value="Agent M" className="bg-(--color-background)">Agent M</option>
                        <option value="Agent Q" className="bg-(--color-background)">Agent Q</option>
                        <option value="Agent O" className="bg-(--color-background)">Agent O</option>
                        <option value="Agent E" className="bg-(--color-background)">Agent E</option>
                        <option value="Ticker Dashboard" className="bg-(--color-background)">Ticker Dashboard</option>
                      </select>
                      <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2">
                        <svg className="h-4 w-4 fill-current text-(--color-neutral-500)" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                        </svg>
                      </div>
                    </div>
                  </div>
                </div>

                {selectedAgent && AGENT_SUBCOMPONENTS[selectedAgent as keyof typeof AGENT_SUBCOMPONENTS].length > 0 && (
                  <div className="flex items-center gap-4">
                    <label 
                      htmlFor="subcomponent"
                      className="w-24 font-medium text-(--color-neutral-100)"
                    >
                      Sub-component:
                    </label>
                    <div className="relative flex-1">
                      <select
                        id="subcomponent"
                        value={selectedSubComponent}
                        onChange={(e) => setSelectedSubComponent(e.target.value)}
                        className="w-full rounded-md border border-[#2A2E39] bg-(--color-background) px-3 py-2 text-(--color-neutral-100) focus:outline-none focus:border-(--color-accent) appearance-none cursor-pointer"
                      >
                        <option value="" className="bg-(--color-background)">Select a sub-component...</option>
                        {AGENT_SUBCOMPONENTS[selectedAgent as keyof typeof AGENT_SUBCOMPONENTS].map((subComponent) => (
                          <option key={subComponent} value={subComponent} className="bg-(--color-background)">
                            {subComponent}
                          </option>
                        ))}
                      </select>
                      <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2">
                        <svg className="h-4 w-4 fill-current text-(--color-neutral-500)" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                        </svg>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Report Variables Section */}
              {selectedAgent && (
                <div className="rounded-lg bg-(--color-background) p-6 shadow-lg relative overflow-hidden border border-[#2A2E39]"
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
                      onRunQA={handleRunQA}
                      isLoading={isLoading}
                      hasActiveQA={hasActiveQA}
                    />
                  </div>
                </div>
              )}

              {/* Error Message */}
              {error && (
                <div className="rounded-md bg-red-500 bg-opacity-10 border border-red-500 p-4 text-red-500">
                  {error}
                </div>
              )}

              {/* QA Response Section */}
              <div ref={qaResponseRef}>
                {qaRun && (
                  <QAResponse
                    qaRun={qaRun}
                    onQARating={handleQARating}
                    onReportRating={handleReportRating}
                    onSubmit={handleSubmit}
                    onRetry={handleRetry}
                  />
                )}
              </div>
            </>
          ) : selectedTab === 'analytics' ? (
            <Analytics />
          ) : selectedTab === 'automated' ? (
            <AutomatedQA />
          ) : (
            <AutomatedAnalytics />
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
