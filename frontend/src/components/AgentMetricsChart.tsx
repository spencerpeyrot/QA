import React, { useState } from 'react';
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
  const startDate = new Date('2024-04-27');
  return Array.from({ length: numPoints }, (_, i) => {
    const currentDate = new Date(startDate);
    currentDate.setDate(startDate.getDate() + i);
    const formattedDate = `${currentDate.getMonth() + 1}/${currentDate.getDate()}`;
    return {
      date: formattedDate,
      rating: Math.random() * 2 + 3, // Random rating between 3-5
    };
  });
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

const getAgentColor = (agentName: string): string => {
  const colorMap: { [key: string]: string } = {
    'Agent S': 'var(--color-agentS-light)',
    'Agent M': 'var(--color-agentM-light)',
    'Agent O': 'var(--color-agentO-light)',
    'Agent E': 'var(--color-agentE-light)',
    'Agent Q': 'var(--color-agentQ-light)',
  };
  return colorMap[agentName] || 'var(--color-accent)';
};

const CustomTooltip = ({ active, payload, agentColor }: any) => {
  if (active && payload && payload.length) {
    const ratingPayload = payload.find((p: any) => p.dataKey === 'rating' && p.name === 'Rating');
    if (ratingPayload) {
      return (
        <div className="bg-[#1a1d24] p-[4px_8px] rounded-lg border border-[#2A2E39] shadow-sm">
          <p style={{ color: agentColor, fontSize: 11 }}>
            {`${ratingPayload.value.toFixed(2)}`}
          </p>
        </div>
      );
    }
  }
  return null;
};

const AgentMetricsChart: React.FC = () => {
  const [selectedTimeFrame, setSelectedTimeFrame] = useState<string>('7D');
  
  // Define the desired order of agents
  const agentOrder = ['Agent S', 'Agent M', 'Agent O', 'Agent E', 'Agent Q'];
  
  // Group components by agent
  const componentsByAgent = components.reduce((acc, component) => {
    if (!acc[component.agent]) {
      acc[component.agent] = [];
    }
    acc[component.agent].push(component);
    return acc;
  }, {} as Record<string, typeof components>);

  // Sort the entries based on the agentOrder array
  const sortedAgentEntries = Object.entries(componentsByAgent).sort(([agentA], [agentB]) => {
    return agentOrder.indexOf(agentA) - agentOrder.indexOf(agentB);
  });

  return (
    <div 
      className="rounded-lg bg-(--color-background) p-6 shadow-lg relative overflow-hidden border border-[#2A2E39]"
      style={{
        backgroundImage: 'radial-gradient(circle, #c1ff0005 1px, transparent 1px)',
        backgroundSize: '4px 4px'
      }}
    >
      <div className="metrics-header flex justify-between items-center">
        <h2 className="text-lg font-medium text-(--color-neutral-100)">Performance Metrics</h2>
        <div className="flex items-center gap-2 text-sm">
          {['7D', '1M', '1Y', 'YTD', 'ALL'].map((timeFrame) => (
            <button
              key={timeFrame}
              onClick={() => setSelectedTimeFrame(timeFrame)}
              className={`px-3 py-1 rounded-md transition-colors ${
                selectedTimeFrame === timeFrame
                  ? 'bg-[--color-neutral-900]/80 border border-(--color-accent) text-(--color-accent)'
                  : 'bg-[--color-neutral-900]/40 text-(--color-neutral-100) hover:bg-[--color-neutral-900]/60'
              }`}
            >
              {timeFrame}
            </button>
          ))}
        </div>
      </div>
      <div className="metrics-content pt-8">
        <div className="metrics-agents-container flex flex-col [&>*+*]:pt-8">
          {sortedAgentEntries.map(([agent, agentComponents]) => (
            <div key={agent} className="flex flex-col gap-4">
              <h3 className="text-md font-medium text-(--color-neutral-100)">{agent}</h3>
              <div className="relative">
                <div className="flex overflow-x-auto pb-4 gap-4 snap-x snap-mandatory [scrollbar-width:none] [-ms-overflow-style:none] [&::-webkit-scrollbar]:hidden">
                  {agentComponents.map((component, idx) => {
                    const agentColor = getAgentColor(agent);
                    const gradientId = `colorRating-${agent}-${idx}`.replace(/\s+/g, '-');
                    
                    return (
                      <div 
                        key={component.name} 
                        className="bg-(--color-background) p-4 rounded-lg border border-[#2A2E39] shadow-sm flex-none w-[300px] snap-start"
                        style={{ position: 'relative', zIndex: 1 }}
                      >
                        <h4 className="text-sm font-medium text-(--color-neutral-100) mb-2">
                          {component.name === agent ? 'Performance' : component.name}
                        </h4>
                        <div className="h-[150px]">
                          <ResponsiveContainer width="100%" height="100%">
                            <ComposedChart data={component.data}>
                              <defs>
                                <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                                  <stop offset="5%" stopColor={agentColor} stopOpacity={0.3}/>
                                  <stop offset="95%" stopColor={agentColor} stopOpacity={0}/>
                                </linearGradient>
                              </defs>
                              <CartesianGrid strokeDasharray="3 3" stroke="#2A2E39" />
                              <XAxis 
                                dataKey="date"
                                tick={{ fontSize: 9, fill: 'var(--color-neutral-100)' }}
                                stroke="#2A2E39"
                              />
                              <YAxis 
                                domain={[0, 5]}
                                tick={{ fontSize: 9, fill: 'var(--color-neutral-100)' }}
                                stroke="#2A2E39"
                                width={40}
                                label={{ 
                                  value: 'Rating', 
                                  angle: -90, 
                                  position: 'insideLeft',
                                  offset: 10,
                                  style: { textAnchor: 'middle', fill: 'var(--color-neutral-100)', fontSize: 9 }
                                }}
                              />
                              <Tooltip 
                                cursor={false}
                                offset={10}
                                position={{ y: -10 }}
                                content={<CustomTooltip agentColor={agentColor} />}
                              />
                              <Area
                                type="monotone"
                                dataKey="rating"
                                stroke="none"
                                fillOpacity={1}
                                fill={`url(#${gradientId})`}
                                activeDot={false}
                                isAnimationActive={false}
                                name=" "
                              />
                              <Line
                                type="monotone"
                                dataKey="rating"
                                stroke={agentColor}
                                strokeWidth={2}
                                dot={false}
                                activeDot={{ r: 2.5, fill: agentColor }}
                                isAnimationActive={false}
                                name="Rating"
                              />
                            </ComposedChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    );
                  })}
                </div>
                {agentComponents.length > 1 && (
                  <div className="absolute -bottom-2 left-0 right-0 flex justify-center gap-1.5">
                    {agentComponents.map((component, index) => (
                      <div
                        key={index}
                        className="w-1.5 h-1.5 rounded-full bg-[--color-neutral-700]"
                        aria-hidden="true"
                      />
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default AgentMetricsChart; 