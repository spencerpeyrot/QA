import React, { useState, useEffect } from 'react';
import { Play, CheckCircle, XCircle, Clock, Loader2 } from 'lucide-react';
import { pipelineApi } from '../services/pipelineApi';

// Dummy data structure for pipeline status
interface PipelineStatus {
  status: 'not_started' | 'in_progress' | 'completed' | 'failed';
  lastRun: string;
}

interface PipelineData {
  [agent: string]: {
    [subComponent: string]: PipelineStatus;
  };
}

// Dummy data
const PIPELINE_DATA: PipelineData = {
  'Home Page': {
    'Ticker Pulse': {
      status: 'completed',
      lastRun: '2024-04-22 16:45:00'
    }
  },
  'Agent S': {
    'Main': {
      status: 'completed',
      lastRun: '2024-04-22 14:30:00'
    }
  },
  'Agent M': {
    'Long Term View': {
      status: 'in_progress',
      lastRun: '2024-04-22 15:45:00'
    },
    'Short Term View': {
      status: 'not_started',
      lastRun: '2024-04-21 09:15:00'
    },
    'Sector Level View': {
      status: 'completed',
      lastRun: '2024-04-22 11:20:00'
    },
    'Key Gamma Levels': {
      status: 'failed',
      lastRun: '2024-04-22 13:10:00'
    }
  },
  'Agent Q': {
    'Support & Resistance': {
      status: 'completed',
      lastRun: '2024-04-22 16:00:00'
    },
    'Volume Analysis': {
      status: 'not_started',
      lastRun: '2024-04-21 14:30:00'
    },
    'Candlestick Patterns': {
      status: 'completed',
      lastRun: '2024-04-22 10:45:00'
    },
    'Consolidated Analysis': {
      status: 'in_progress',
      lastRun: '2024-04-22 16:30:00'
    }
  },
  'Agent O': {
    'Flow Analysis': {
      status: 'completed',
      lastRun: '2024-04-22 12:15:00'
    },
    'Volatility Analysis': {
      status: 'not_started',
      lastRun: '2024-04-21 16:45:00'
    },
    'Consolidated Analysis': {
      status: 'failed',
      lastRun: '2024-04-22 14:20:00'
    }
  },
  'Agent E': {
    'EPS/Revenue Analysis': {
      status: 'completed',
      lastRun: '2024-04-22 15:00:00'
    },
    'Forward Guidance': {
      status: 'in_progress',
      lastRun: '2024-04-22 16:15:00'
    },
    'Quantitative Analysis': {
      status: 'not_started',
      lastRun: '2024-04-21 11:30:00'
    },
    'Options Analysis': {
      status: 'completed',
      lastRun: '2024-04-22 13:45:00'
    },
    'Consolidated Analysis': {
      status: 'failed',
      lastRun: '2024-04-22 15:30:00'
    }
  }
};

const StatusIcon = ({ status }: { status: PipelineStatus['status'] }) => {
  switch (status) {
    case 'completed':
      return <CheckCircle className="text-green-500" size={20} />;
    case 'failed':
      return <XCircle className="text-red-500" size={20} />;
    case 'in_progress':
      return <Clock className="text-yellow-500 animate-pulse" size={20} />;
    default:
      return <div className="w-5 h-5 rounded-full border-2 border-(--color-neutral-500)" />;
  }
};

const PipelineRow = ({ 
  agent, 
  subComponent, 
  status, 
  lastRun,
  isRunning,
  onRunPipeline
}: { 
  agent: string; 
  subComponent: string; 
  status: PipelineStatus['status']; 
  lastRun: string;
  isRunning: boolean;
  onRunPipeline: (agent: string, subComponent: string) => Promise<void>;
}) => {
  const handleRunPipeline = async () => {
    await onRunPipeline(agent, subComponent);
  };

  return (
    <div className="flex items-center justify-between py-3 px-4 border-b border-[#2A2E39] hover:bg-[#1a1d24] transition-colors">
      <div className="flex items-center gap-4 flex-1">
        {isRunning ? (
          <Loader2 className="text-yellow-500 animate-spin" size={20} />
        ) : (
          <StatusIcon status={status} />
        )}
        <div>
          <div className="font-medium text-(--color-neutral-100)">{subComponent}</div>
          <div className="text-sm text-(--color-neutral-500)">{agent}</div>
        </div>
      </div>
      <div className="flex items-center gap-6">
        <div className="text-sm text-(--color-neutral-500)">
          Last run: {new Date(lastRun).toLocaleString()}
        </div>
        <button
          onClick={handleRunPipeline}
          disabled={isRunning}
          className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-transparent border border-(--color-accent) text-(--color-accent) font-medium text-sm hover:bg-(--color-accent) hover:text-black transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isRunning ? (
            <>
              <Loader2 className="animate-spin" size={16} />
              Running...
            </>
          ) : (
            <>
              <Play size={16} />
              Run Pipeline
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export function AutomatedQA() {
  const [activeRuns, setActiveRuns] = useState<{ [key: string]: boolean }>({});
  const [error, setError] = useState<string | null>(null);
  const [pipelineStatus, setPipelineStatus] = useState<'not_started' | 'running' | 'completed' | 'failed'>('not_started');

  // Poll for status when pipeline is running
  useEffect(() => {
    let intervalId: NodeJS.Timeout;

    const pollStatus = async () => {
      try {
        const status = await pipelineApi.getPipelineStatus();
        if (status.status === 'completed' || status.status === 'failed') {
          setActiveRuns({});
          setPipelineStatus(status.status);
          if (status.error) {
            setError(status.error);
          }
        }
      } catch (err: any) {
        console.error('Error polling status:', err);
      }
    };

    if (Object.values(activeRuns).some(Boolean)) {
      intervalId = setInterval(pollStatus, 5000); // Poll every 5 seconds
    }

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [activeRuns]);

  const handleRunPipeline = async (agent: string, subComponent: string) => {
    const runKey = `${agent}-${subComponent}`;
    
    try {
      setError(null);
      setPipelineStatus('running');
      setActiveRuns(prev => ({ ...prev, [runKey]: true }));

      // Handle different pipeline types
      if (agent === 'Agent M' && subComponent === 'Long Term View') {
        const response = await pipelineApi.runLTVPipeline(agent, subComponent);
        console.log('LTV Pipeline run response:', response);
      } else if (agent === 'Home Page' && subComponent === 'Ticker Pulse') {
        const response = await pipelineApi.runTickerPulsePipeline(agent, subComponent);
        console.log('Ticker Pulse Pipeline run response:', response);
      } else {
        throw new Error('This pipeline is not yet implemented');
      }
    } catch (err: any) {
      setError(err.message || 'Failed to run pipeline');
      setPipelineStatus('failed');
      console.error('Pipeline run error:', err);
      setActiveRuns(prev => ({ ...prev, [runKey]: false }));
    }
  };

  return (
    <div className="space-y-8 m-6 [&>*]:!mb-8">
      {error && (
        <div className="rounded-md bg-red-500 bg-opacity-10 border border-red-500 p-4 text-red-500">
          {error}
        </div>
      )}
      {Object.entries(PIPELINE_DATA).map(([agent, subComponents]) => (
        <div 
          key={agent}
          className="rounded-lg bg-(--color-background) shadow-lg relative overflow-hidden border border-[#2A2E39] !mt-0"
          style={{
            backgroundImage: 'radial-gradient(circle, #c1ff0005 1px, transparent 1px)',
            backgroundSize: '4px 4px'
          }}
        >
          <div className="px-6 py-4 border-b border-[#2A2E39]">
            <h2 className="text-lg font-medium text-(--color-neutral-100)">{agent}</h2>
          </div>
          <div>
            {Object.entries(subComponents).map(([subComponent, { status, lastRun }]) => (
              <PipelineRow
                key={`${agent}-${subComponent}`}
                agent={agent}
                subComponent={subComponent}
                status={status}
                lastRun={lastRun}
                isRunning={activeRuns[`${agent}-${subComponent}`] || false}
                onRunPipeline={handleRunPipeline}
              />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
} 