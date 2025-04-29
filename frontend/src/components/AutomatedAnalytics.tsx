import React from 'react';
import { BarChart3, TrendingUp, TrendingDown, AlertTriangle } from 'lucide-react';

interface ComponentStats {
  successRate: number;
  averageRunTime: number;
  totalRuns: number;
  failureRate: number;
  lastWeekTrend: 'up' | 'down' | 'stable';
  commonErrors: string[];
}

interface AgentStats {
  [subComponent: string]: ComponentStats;
}

// Dummy data for component statistics
const AGENT_STATS: Record<string, AgentStats> = {
  'Agent S': {
    'Main': {
      successRate: 95.5,
      averageRunTime: 45,
      totalRuns: 1250,
      failureRate: 4.5,
      lastWeekTrend: 'up',
      commonErrors: ['Timeout', 'Data validation error']
    }
  },
  'Agent M': {
    'Long Term View': {
      successRate: 88.2,
      averageRunTime: 120,
      totalRuns: 980,
      failureRate: 11.8,
      lastWeekTrend: 'down',
      commonErrors: ['API rate limit', 'Data inconsistency']
    },
    'Short Term View': {
      successRate: 92.7,
      averageRunTime: 60,
      totalRuns: 1100,
      failureRate: 7.3,
      lastWeekTrend: 'stable',
      commonErrors: ['Missing market data']
    },
    'Sector Level View': {
      successRate: 94.1,
      averageRunTime: 90,
      totalRuns: 850,
      failureRate: 5.9,
      lastWeekTrend: 'up',
      commonErrors: ['Incomplete sector data']
    },
    'Key Gamma Levels': {
      successRate: 91.5,
      averageRunTime: 75,
      totalRuns: 920,
      failureRate: 8.5,
      lastWeekTrend: 'down',
      commonErrors: ['Calculation error', 'Invalid input data']
    }
  },
  'Agent Q': {
    'Support & Resistance': {
      successRate: 96.3,
      averageRunTime: 30,
      totalRuns: 1500,
      failureRate: 3.7,
      lastWeekTrend: 'up',
      commonErrors: ['Pattern recognition error']
    },
    'Volume Analysis': {
      successRate: 97.8,
      averageRunTime: 25,
      totalRuns: 1600,
      failureRate: 2.2,
      lastWeekTrend: 'stable',
      commonErrors: ['Data source unavailable']
    },
    'Candlestick Patterns': {
      successRate: 98.1,
      averageRunTime: 20,
      totalRuns: 1800,
      failureRate: 1.9,
      lastWeekTrend: 'up',
      commonErrors: ['Pattern validation failed']
    },
    'Consolidated Analysis': {
      successRate: 93.4,
      averageRunTime: 150,
      totalRuns: 750,
      failureRate: 6.6,
      lastWeekTrend: 'down',
      commonErrors: ['Integration error', 'Timeout']
    }
  }
};

const TrendIcon = ({ trend }: { trend: ComponentStats['lastWeekTrend'] }) => {
  switch (trend) {
    case 'up':
      return <TrendingUp className="text-green-500" size={20} />;
    case 'down':
      return <TrendingDown className="text-red-500" size={20} />;
    default:
      return <BarChart3 className="text-(--color-neutral-500)" size={20} />;
  }
};

const StatCard = ({ label, value, unit = '' }: { label: string; value: number; unit?: string }) => (
  <div className="bg-[#1a1d24] rounded-lg p-4">
    <div className="text-sm text-(--color-neutral-500) mb-1">{label}</div>
    <div className="text-xl font-medium text-(--color-neutral-100)">
      {value.toLocaleString()}{unit}
    </div>
  </div>
);

const ComponentStatsCard = ({ stats }: { stats: ComponentStats }) => (
  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
    <StatCard label="Success Rate" value={stats.successRate} unit="%" />
    <StatCard label="Average Run Time" value={stats.averageRunTime} unit="s" />
    <StatCard label="Total Runs" value={stats.totalRuns} />
    <StatCard label="Failure Rate" value={stats.failureRate} unit="%" />
  </div>
);

const ErrorList = ({ errors }: { errors: string[] }) => (
  <div className="mt-4">
    <div className="text-sm font-medium text-(--color-neutral-100) mb-2 flex items-center gap-2">
      <AlertTriangle size={16} className="text-yellow-500" />
      Common Errors
    </div>
    <ul className="list-disc list-inside text-sm text-(--color-neutral-500)">
      {errors.map((error, index) => (
        <li key={index}>{error}</li>
      ))}
    </ul>
  </div>
);

export function AutomatedAnalytics() {
  return (
    <div className="space-y-8 m-6 [&>*]:!mb-8">
      {Object.entries(AGENT_STATS).map(([agent, subComponents]) => (
        <div 
          key={agent}
          className="rounded-lg bg-(--color-background) shadow-lg relative overflow-hidden border border-[#2A2E39]"
          style={{
            backgroundImage: 'radial-gradient(circle, #c1ff0005 1px, transparent 1px)',
            backgroundSize: '4px 4px'
          }}
        >
          <div className="px-6 py-4 border-b border-[#2A2E39]">
            <h2 className="text-lg font-medium text-(--color-neutral-100)">{agent}</h2>
          </div>
          <div className="p-6 space-y-8">
            {Object.entries(subComponents).map(([subComponent, stats]) => (
              <div key={subComponent} className="space-y-2">
                <div className="flex items-center justify-between">
                  <h3 className="text-md font-medium text-(--color-neutral-100)">
                    {subComponent}
                  </h3>
                  <div className="flex items-center gap-2">
                    <TrendIcon trend={stats.lastWeekTrend} />
                    <span className="text-sm text-(--color-neutral-500)">
                      Last 7 days
                    </span>
                  </div>
                </div>
                <ComponentStatsCard stats={stats} />
                <ErrorList errors={stats.commonErrors} />
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
} 