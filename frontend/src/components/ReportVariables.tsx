import { useState, useEffect } from 'react';

interface ReportVariablesProps {
  agent: string;
  subComponent: string;
  onVariablesChange: (variables: Record<string, string>) => void;
}

type AgentVariables = {
  [K in keyof typeof REQUIRED_VARIABLES]: {
    [SubComponent in string]: string[];
  };
};

// Define the variables required for each agent/sub-component combination
const REQUIRED_VARIABLES = {
  'S': {
    '': ['current_date', 'report_text']
  },
  'M': {
    'Long Term View': ['current_date', 'category', 'report_text'],
    'Short Term View': ['current_date', 'prev_trading_day', 'market_recap', 'market_thesis'],
    'Sector Level View': ['current_date', 'sector_type', 'report_text'],
    'Key Gamma Levels': ['current_date', 'ticker', 'report_text']
  },
  'Q': {
    'Support & Resistance': ['current_date', 'ticker', 'current_price', 'report_text'],
    'Volume Analysis': ['current_date', 'ticker', 'report_text'],
    'Candlestick Patterns': ['current_date', 'ticker', 'report_text'],
    'Consolidated Analysis': ['current_date', 'ticker', 'current_price', 'report_text']
  },
  'O': {
    'Flow Analysis': ['current_date', 'ticker', 'current_price', 'report_text'],
    'Volatility Analysis': ['current_date', 'ticker', 'current_price', 'report_text'],
    'Consolidated Analysis': ['current_date', 'ticker', 'current_price', 'report_text']
  },
  'E': {
    'EPS/Revenue Analysis': ['current_date', 'ticker', 'report_text'],
    'Forward Guidance': ['current_date', 'ticker', 'report_text'],
    'Quantitative Analysis': ['current_date', 'ticker', 'report_text'],
    'Options Analysis': ['current_date', 'ticker', 'report_text'],
    'Consolidated Analysis': ['current_date', 'ticker', 'eps_rev_report', 'fg_report', 'quant_report', 'options_report']
  },
  'TD': {
    'Overview': ['ticker', 'agent_ratings', 'overview_report'],
    'Agent S Snapshot': ['current_date', 'ticker', 'report_text'],
    'Agent Q Snapshot': ['current_date', 'ticker', 'current_price', 'td_q_report', 'q_standalone_report'],
    'Agent O Snapshot': ['current_date', 'ticker', 'current_price', 'td_o_report', 'o_standalone_report'],
    'Agent M Snapshot': ['current_date', 'ticker', 'report_text']
  }
} as const;

// Helper function to get human-readable label
const getFieldLabel = (field: string): string => {
  const labels: Record<string, string> = {
    current_date: 'Current Date',
    report_text: 'Report Text',
    ticker: 'Ticker Symbol',
    current_price: 'Current Price',
    category: 'Category',
    sector_type: 'Sector Type',
    prev_trading_day: 'Previous Trading Day',
    market_recap: 'Market Recap',
    market_thesis: 'Market Thesis',
    agent_ratings: 'Agent Ratings',
    overview_report: 'Overview Report',
    eps_rev_report: 'EPS/Revenue Report',
    fg_report: 'Forward Guidance Report',
    quant_report: 'Quantitative Report',
    options_report: 'Options Report',
    td_q_report: 'Agent Q Report',
    q_standalone_report: 'Q Standalone Report',
    td_o_report: 'Agent O Report',
    o_standalone_report: 'O Standalone Report'
  };
  return labels[field] || field.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
};

export function ReportVariables({ agent, subComponent, onVariablesChange }: ReportVariablesProps) {
  const [variables, setVariables] = useState<Record<string, string>>({});
  
  // Get the required variables for the current agent/sub-component
  const requiredVars = (REQUIRED_VARIABLES as AgentVariables)[agent]?.[subComponent] || [];
  
  // Reset variables when agent or sub-component changes
  useEffect(() => {
    const newVariables: Record<string, string> = {};
    requiredVars.forEach((variable: string) => {
      // Preserve existing values when they exist
      newVariables[variable] = variables[variable] || '';
      
      // Set default value for current_date
      if (variable === 'current_date' && !variables[variable]) {
        newVariables[variable] = new Date().toISOString().split('T')[0];
      }
    });
    setVariables(newVariables);
    onVariablesChange(newVariables);
  }, [agent, subComponent]);

  const handleInputChange = (field: string, value: string) => {
    const newVariables = { ...variables, [field]: value };
    setVariables(newVariables);
    onVariablesChange(newVariables);
  };

  if (!agent || !requiredVars.length) return null;

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-medium text-(--color-neutral-100)">Report Variables</h2>
      <div className="grid gap-4 md:grid-cols-2">
        {requiredVars.map((field: string) => (
          <div key={field} className="space-y-2">
            <label 
              htmlFor={field}
              className="block text-sm font-medium text-(--color-neutral-100)"
            >
              {getFieldLabel(field)}:
            </label>
            {field.includes('report') || field === 'market_recap' || field === 'market_thesis' ? (
              <textarea
                id={field}
                value={variables[field] || ''}
                onChange={(e) => handleInputChange(field, e.target.value)}
                className="w-full rounded-md border border-[#2A2E39] bg-[#1a1d24] px-3 py-2 text-(--color-neutral-100) focus:outline-none focus:border-(--color-accent) min-h-[100px]"
                placeholder={`Enter ${getFieldLabel(field).toLowerCase()}...`}
              />
            ) : (
              <input
                type={field === 'current_date' ? 'date' : field.includes('price') ? 'number' : 'text'}
                id={field}
                value={variables[field] || ''}
                onChange={(e) => handleInputChange(field, e.target.value)}
                className="w-full rounded-md border border-[#2A2E39] bg-[#1a1d24] px-3 py-2 text-(--color-neutral-100) focus:outline-none focus:border-(--color-accent)"
                placeholder={`Enter ${getFieldLabel(field).toLowerCase()}...`}
              />
            )}
          </div>
        ))}
      </div>
    </div>
  );
} 