import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  ComposedChart
} from 'recharts';

// Dummy data structure
const generateDummyData = (numPoints: number) => {
  return Array.from({ length: numPoints }, (_, i) => ({
    date: `Day ${i + 1}`,
    rating: Math.random() * 2 + 3, // Random rating between 3-5
  }));
};

// Actual agent structure from registry.json
const agentStructure = {
  "Agent S": null, // No subcomponents
  "Agent M": [
    "Long Term View",
    "Short Term View",
    "Sector Level View",
    "Key Gamma Levels"
  ],
  "Agent Q": [
    "Support & Resistance",
    "Volume Analysis",
    "Candlestick Patterns",
    "Consolidated Analysis"
  ],
  "Agent O": [
    "Flow Analysis",
    "Volatility Analysis",
    "Consolidated Analysis"
  ],
  "Agent E": [
    "EPS/Revenue Analysis",
    "Forward Guidance",
    "Quantitative Analysis",
    "Options Analysis",
    "Consolidated Analysis"
  ]
};

// Generate components data based on actual structure
const components = Object.entries(agentStructure).flatMap(([agent, subcomponents]) => {
  if (!subcomponents) {
    // For Agent S with no subcomponents
    return [{
      name: agent,
      agent,
      data: generateDummyData(7)
    }];
  }
  // For agents with subcomponents
  return subcomponents.map(subcomponent => ({
    name: subcomponent,
    agent,
    data: generateDummyData(7)
  }));
});

const AgentMetricsChart: React.FC = () => {
  // Group components by agent
  const componentsByAgent = components.reduce((acc, component) => {
    if (!acc[component.agent]) {
      acc[component.agent] = [];
    }
    acc[component.agent].push(component);
    return acc;
  }, {} as Record<string, typeof components>);

  return (
    <div className="rounded-lg bg-(--color-background) p-6 shadow-lg relative overflow-hidden border border-[#2A2E39]">
      <h2 className="text-lg font-medium text-(--color-neutral-100) mb-6">Performance Metrics</h2>
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
        {Object.entries(componentsByAgent).map(([agent, agentComponents]) => (
          <div key={agent} className="flex flex-col gap-4">
            <h3 className="text-md font-medium text-(--color-neutral-100)">{agent}</h3>
            {agentComponents.map((component) => (
              <div 
                key={component.name} 
                className="bg-[--color-neutral-900]/80 p-4 rounded-lg border border-[#2A2E39] shadow-sm"
              >
                <h4 className="text-sm font-medium text-(--color-neutral-100) mb-2">
                  {component.name === agent ? 'Performance' : component.name}
                </h4>
                <div className="h-[200px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={component.data}>
                      <defs>
                        <linearGradient id="colorRating" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="var(--color-accent)" stopOpacity={0.3}/>
                          <stop offset="95%" stopColor="var(--color-accent)" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#2A2E39" />
                      <XAxis 
                        dataKey="date"
                        tick={{ fontSize: 10, fill: 'var(--color-neutral-100)' }}
                        stroke="#2A2E39"
                      />
                      <YAxis 
                        domain={[0, 5]}
                        tick={{ fontSize: 10, fill: 'var(--color-neutral-100)' }}
                        stroke="#2A2E39"
                        label={{ 
                          value: 'Rating', 
                          angle: -90, 
                          position: 'insideLeft',
                          style: { textAnchor: 'middle', fill: 'var(--color-neutral-100)', fontSize: 10 }
                        }}
                      />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: '#1a1d24',
                          border: '1px solid #2A2E39',
                          borderRadius: '4px',
                          color: 'var(--color-neutral-100)',
                          fontSize: 12
                        }}
                      />
                      <Area
                        type="monotone"
                        dataKey="rating"
                        stroke="none"
                        fillOpacity={1}
                        fill="url(#colorRating)"
                      />
                      <Line
                        type="monotone"
                        dataKey="rating"
                        stroke="var(--color-accent)"
                        strokeWidth={2}
                        dot={{ r: 3, fill: 'var(--color-accent)' }}
                        name="Output Rating"
                      />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
};

export default AgentMetricsChart; 