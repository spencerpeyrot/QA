import { useState } from 'react';

const AGENTS = ['S', 'M', 'Q', 'O', 'E', 'Ticker Dashboard'];

interface AgentSelectorProps {
  onAgentSelect: (agent: string) => void;
  onSubComponentSelect: (subComponent: string | null) => void;
}

export function AgentSelector({ onAgentSelect, onSubComponentSelect }: AgentSelectorProps) {
  const [selectedAgent, setSelectedAgent] = useState<string>('');
  
  const handleAgentChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const agent = event.target.value;
    setSelectedAgent(agent);
    onAgentSelect(agent);
    onSubComponentSelect(null); // Reset sub-component when agent changes
  };

  return (
    <div className="space-y-4">
      <div>
        <label htmlFor="agent" className="block text-sm font-medium text-gray-700">
          Agent
        </label>
        <select
          id="agent"
          value={selectedAgent}
          onChange={handleAgentChange}
          className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
        >
          <option value="">Select an agent...</option>
          {AGENTS.map((agent) => (
            <option key={agent} value={agent}>
              Agent {agent}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
} 