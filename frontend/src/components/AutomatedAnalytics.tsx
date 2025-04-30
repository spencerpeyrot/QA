import React, { useEffect, useState } from 'react';
import { BarChart3, TrendingUp, TrendingDown, AlertTriangle, CheckCircle, XCircle } from 'lucide-react';
import { evaluationApi, type AgentStats, type EvaluationMetrics } from '../services/evaluationApi';

interface ComponentStats {
  successRate: number;
  averageRunTime: number;
  totalRuns: number;
  failureRate: number;
  lastWeekTrend: 'up' | 'down' | 'stable';
  commonErrors: string[];
  ltvMetrics?: EvaluationMetrics;
  tickerPulseMetrics?: EvaluationMetrics;
}

interface LTVEvaluation {
  _id: string;
  document_id: string;
  user_question: string;
  timestamp: string;
  evaluated_at: string;
  evaluation: {
    factual_criteria: {
      accurate_numbers: boolean;
      correct_citations: boolean;
    };
    completeness_criteria: {
      covers_macro_context: boolean;
      includes_context: boolean;
    };
    quality_criteria: {
      clear_presentation: boolean;
      explains_causes: boolean;
    };
    hallucination_free: boolean;
    quality_score: number;
    criteria_explanations: {
      accurate_numbers: string;
      correct_citations: string;
      covers_macro_context: string;
      includes_context: string;
      clear_presentation: string;
      explains_causes: string;
      hallucination_free: string;
    };
    unsupported_claims: string[];
    score_explanation: string;
  };
  pipeline: string;
}

interface LTVStats {
  averageQualityScore: number;
  factualAccuracyRate: number;
  completenessRate: number;
  qualityRate: number;
  hallucinationFreeRate: number;
  totalEvaluations: number;
}

// Dummy data for component statistics
const AGENT_STATS: Record<string, AgentStats> = {
  'Agent M': {
    'Long Term View': {
      successRate: 92.5,
      averageRunTime: 120,
      totalRuns: 25,
      failureRate: 7.5,
      lastWeekTrend: 'up',
      commonErrors: [
        'Hallucination detected',
        'Incomplete macro context',
        'Citation accuracy issues',
        'Missing key trends'
      ],
      ltvMetrics: {
        factualAccuracyRate: 92.5,
        completenessRate: 88.0,
        qualityUsefulnessRate: 90.0,
        hallucinationFreeRate: 85.0,
        averageQualityScore: 87.5,
        totalDocumentsEvaluated: 25,
        documentsRequiringCorrection: 5
      }
    }
  },
  'Ticker Pulse': {
    'Main': {
      successRate: 94.0,
      averageRunTime: 45,
      totalRuns: 2200,
      failureRate: 6.0,
      lastWeekTrend: 'up',
      commonErrors: [
        'Missing momentum context',
        'Incomplete market data',
        'Citation accuracy issues'
      ],
      tickerPulseMetrics: {
        factualAccuracyRate: 95.0,
        completenessRate: 92.0,
        qualityUsefulnessRate: 88.0,
        hallucinationFreeRate: 94.0,
        averageQualityScore: 78.0,
        totalDocumentsEvaluated: 2200,
        documentsRequiringCorrection: 132  // Assuming 6% failure rate of total documents
      }
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

const StatCardSkeleton = () => {
  return (
    <div className="bg-[#1a1d24] rounded-xl border border-[#2A2E39] p-6 animate-pulse">
      <div className="flex flex-col gap-2">
        <div className="h-4 w-24 bg-[#2A2E39] rounded"></div>
        <div className="flex items-baseline gap-2">
          <div className="h-8 w-16 bg-[#2A2E39] rounded"></div>
          <div className="h-4 w-8 bg-[#2A2E39] rounded"></div>
        </div>
      </div>
    </div>
  );
};

const StatCard = ({ label, value, unit }: { label: string; value: number; unit?: string }) => {
  const formattedValue = unit === '%' 
    ? value.toFixed(1)
    : value.toLocaleString();

  return (
    <div className="bg-[#1a1d24] rounded-xl border border-[#2A2E39] p-6 hover:border-[#3A3E49] transition-all">
      <div className="flex flex-col gap-2">
        <div className="text-sm font-medium text-(--color-neutral-500)">{label}</div>
        <div className="flex items-baseline gap-2">
          <div className="text-3xl font-semibold text-(--color-neutral-100)">
            {formattedValue}
          </div>
          {unit && (
            <div className="text-sm font-medium text-(--color-neutral-500)">
              {unit}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

const MetricsGrid = ({ metrics, title }: { metrics: EvaluationMetrics; title: string }) => {
  const hasData = metrics.totalDocumentsEvaluated > 0;

  if (!hasData) {
    return (
      <div className="bg-white rounded-lg p-6 mt-6">
        <h3 className="text-lg font-medium mb-4">{title}</h3>
        <div className="text-gray-500 text-center py-8">
          No evaluation data available yet. Statistics will appear here once evaluations are processed.
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg p-6 mt-6">
      <h3 className="text-lg font-medium mb-4">{title}</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        <StatCard 
          label="Factual Accuracy Rate" 
          value={metrics.factualAccuracyRate} 
          unit="%" 
        />
        <StatCard 
          label="Completeness Rate" 
          value={metrics.completenessRate} 
          unit="%" 
        />
        <StatCard 
          label="Quality/Usefulness Rate" 
          value={metrics.qualityUsefulnessRate} 
          unit="%" 
        />
        <StatCard 
          label="Hallucination-Free Rate" 
          value={metrics.hallucinationFreeRate} 
          unit="%" 
        />
        <StatCard 
          label="Average Quality Score" 
          value={metrics.averageQualityScore} 
          unit="/10" 
        />
        <StatCard 
          label="Total Documents" 
          value={metrics.totalDocumentsEvaluated} 
        />
        <StatCard 
          label="Documents Needing Correction" 
          value={metrics.documentsRequiringCorrection} 
        />
      </div>
    </div>
  );
};

const ComponentStatsCard = ({ stats }: { stats: ComponentStats }) => {
  if (stats.ltvMetrics) {
    return <MetricsGrid metrics={stats.ltvMetrics} title="LTV Metrics" />;
  }
  
  if (stats.tickerPulseMetrics) {
    return <MetricsGrid metrics={stats.tickerPulseMetrics} title="Ticker Pulse Metrics" />;
  }

  // Default layout for non-metrics components
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 my-6">
      <StatCard label="Success Rate" value={stats.successRate} unit="%" />
      <StatCard label="Average Run Time" value={stats.averageRunTime} unit="s" />
      <StatCard label="Total Runs" value={stats.totalRuns} />
      <StatCard label="Failure Rate" value={stats.failureRate} unit="%" />
    </div>
  );
};

const ErrorList = ({ errors }: { errors: string[] }) => (
  <div className="mt-6 mb-2">
    <div className="text-sm font-medium text-(--color-neutral-100) mb-4 flex items-center gap-2">
      <AlertTriangle size={16} className="text-yellow-500" />
      Common Errors
    </div>
    <ul className="list-disc list-inside text-sm text-(--color-neutral-500) space-y-1">
      {errors.map((error, index) => (
        <li key={index}>{error}</li>
      ))}
    </ul>
  </div>
);

const LTVEvaluationTable: React.FC = () => {
  const [evaluations, setEvaluations] = useState<LTVEvaluation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set());

  useEffect(() => {
    const fetchEvaluations = async () => {
      try {
        const response = await fetch('http://localhost:8000/evaluations/ltv');
        console.log('Response status:', response.status);
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status} ${response.statusText}`);
        }
        
        const data = await response.json();
        console.log('Received data:', data);
        setEvaluations(data);
      } catch (err) {
        console.error('Error fetching evaluations:', err);
        setError(err instanceof Error ? err.message : 'Failed to fetch evaluations');
      } finally {
        setLoading(false);
      }
    };

    fetchEvaluations();
  }, []);

  const toggleRow = (documentId: string) => {
    setExpandedRows(prev => {
      const next = new Set(prev);
      if (next.has(documentId)) {
        next.delete(documentId);
      } else {
        next.add(documentId);
      }
      return next;
    });
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-md bg-red-50 p-4 my-4">
        <div className="flex">
          <AlertTriangle className="h-5 w-5 text-red-400" />
          <div className="ml-3">
            <h3 className="text-sm font-medium text-red-800">Error loading evaluations</h3>
            <p className="text-sm text-red-700 mt-1">{error}</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="overflow-hidden shadow ring-1 ring-black ring-opacity-5 sm:rounded-lg">
      <table className="min-w-full divide-y divide-gray-300">
        <thead className="bg-gray-50">
          <tr>
            <th scope="col" className="py-3.5 pl-4 pr-3 text-left text-sm font-semibold text-gray-900 sm:pl-6">Document ID</th>
            <th scope="col" className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">Question</th>
            <th scope="col" className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">Evaluated</th>
            <th scope="col" className="px-3 py-3.5 text-center text-sm font-semibold text-gray-900">Factual</th>
            <th scope="col" className="px-3 py-3.5 text-center text-sm font-semibold text-gray-900">Complete</th>
            <th scope="col" className="px-3 py-3.5 text-center text-sm font-semibold text-gray-900">Quality</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-200 bg-white">
          {evaluations.map((evaluation) => (
            <React.Fragment key={evaluation.document_id}>
              <tr 
                className="hover:bg-gray-50 cursor-pointer"
                onClick={() => toggleRow(evaluation.document_id)}
              >
                <td className="whitespace-nowrap py-4 pl-4 pr-3 text-sm font-medium text-gray-900 sm:pl-6">
                  {evaluation.document_id}
                </td>
                <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">
                  {evaluation.user_question}
                </td>
                <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">
                  {new Date(evaluation.evaluated_at).toLocaleDateString()}
                </td>
                <td className="whitespace-nowrap px-3 py-4 text-sm text-center">
                  <StatusCell value={evaluation.evaluation.factual_criteria.accurate_numbers && 
                                 evaluation.evaluation.factual_criteria.correct_citations} />
                </td>
                <td className="whitespace-nowrap px-3 py-4 text-sm text-center">
                  <StatusCell value={evaluation.evaluation.completeness_criteria.covers_macro_context && 
                                 evaluation.evaluation.completeness_criteria.includes_context} />
                </td>
                <td className="whitespace-nowrap px-3 py-4 text-sm text-center">
                  <StatusCell value={evaluation.evaluation.quality_criteria.clear_presentation && 
                                 evaluation.evaluation.quality_criteria.explains_causes} />
                </td>
              </tr>
              {expandedRows.has(evaluation.document_id) && (
                <tr className="bg-gray-50">
                  <td colSpan={6} className="px-6 py-4">
                    <div className="space-y-4">
                      <div>
                        <h4 className="text-sm font-medium text-gray-900">Factual Accuracy Explanation</h4>
                        <p className="mt-1 text-sm text-gray-600">{evaluation.evaluation.criteria_explanations.accurate_numbers}</p>
                      </div>
                      <div>
                        <h4 className="text-sm font-medium text-gray-900">Completeness Explanation</h4>
                        <p className="mt-1 text-sm text-gray-600">{evaluation.evaluation.criteria_explanations.correct_citations}</p>
                      </div>
                      <div>
                        <h4 className="text-sm font-medium text-gray-900">Quality/Usefulness Explanation</h4>
                        <p className="mt-1 text-sm text-gray-600">{evaluation.evaluation.criteria_explanations.covers_macro_context}</p>
                      </div>
                      {evaluation.evaluation.unsupported_claims && evaluation.evaluation.unsupported_claims.length > 0 && (
                        <div>
                          <h4 className="text-sm font-medium text-gray-900">Unsupported Claims</h4>
                          <ul className="mt-1 list-disc list-inside text-sm text-gray-600">
                            {evaluation.evaluation.unsupported_claims.map((claim, idx) => (
                              <li key={idx}>{claim}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  </td>
                </tr>
              )}
            </React.Fragment>
          ))}
        </tbody>
      </table>
    </div>
  );
};

interface StatusCellProps {
  value: boolean;
}

const StatusCell: React.FC<StatusCellProps> = ({ value }) => {
  return (
    <span
      className={`inline-flex items-center rounded-full px-2 py-1 text-xs font-medium ${
        value
          ? 'bg-green-50 text-green-700 ring-1 ring-inset ring-green-600/20'
          : 'bg-red-50 text-red-700 ring-1 ring-inset ring-red-600/20'
      }`}
    >
      {value ? (
        <>
          <CheckCircle className="mr-1 h-3 w-3" />
          Pass
        </>
      ) : (
        <>
          <XCircle className="mr-1 h-3 w-3" />
          Fail
        </>
      )}
    </span>
  );
};

const AutomatedAnalytics = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [ltvStats, setLtvStats] = useState<LTVStats | null>(null);
  const [tickerPulseStats, setTickerPulseStats] = useState<LTVStats | null>(null);

  useEffect(() => {
    const fetchStats = async () => {
      setLoading(true);
      try {
        // Fetch LTV stats
        const ltvResponse = await fetch('http://localhost:8000/evaluations/ltv/stats');
        if (!ltvResponse.ok) {
          throw new Error(`Failed to fetch LTV stats: ${ltvResponse.status} ${ltvResponse.statusText}`);
        }
        const ltvData = await ltvResponse.json();
        setLtvStats(ltvData);

        // Fetch Ticker Pulse stats
        const tickerPulseResponse = await fetch('http://localhost:8000/evaluations/ticker-pulse/stats');
        if (!tickerPulseResponse.ok) {
          throw new Error(`Failed to fetch Ticker Pulse stats: ${tickerPulseResponse.status} ${tickerPulseResponse.statusText}`);
        }
        const tickerPulseData = await tickerPulseResponse.json();
        setTickerPulseStats(tickerPulseData);
      } catch (error) {
        console.error('Error fetching stats:', error);
        setError(error instanceof Error ? error.message : 'An error occurred');
      } finally {
        setLoading(false);
      }
    };

    fetchStats();
  }, []);

  if (error) {
    return (
      <div className="rounded-md bg-red-500 bg-opacity-10 border border-red-500 p-4 text-red-500 m-6">
        {error}
      </div>
    );
  }

  const renderStatCards = (stats: LTVStats | null) => {
    if (loading) {
      return (
        <>
          <StatCardSkeleton />
          <StatCardSkeleton />
          <StatCardSkeleton />
          <StatCardSkeleton />
          <StatCardSkeleton />
        </>
      );
    }

    if (!stats) return null;

    return (
      <>
        <StatCard
          label="Quality Score"
          value={stats.averageQualityScore || 0}
          unit="/100"
        />
        <StatCard
          label="Factual Accuracy"
          value={stats.factualAccuracyRate || 0}
          unit="%"
        />
        <StatCard
          label="Completeness"
          value={stats.completenessRate || 0}
          unit="%"
        />
        <StatCard
          label="Quality Pass Rate"
          value={stats.qualityRate || 0}
          unit="%"
        />
        <StatCard
          label="Hallucination-Free"
          value={stats.hallucinationFreeRate || 0}
          unit="%"
        />
      </>
    );
  };

  return (
    <div className="space-y-8 m-6">
      <div 
        className="rounded-lg bg-(--color-background) shadow-lg relative overflow-hidden border border-[#2A2E39]"
        style={{
          backgroundImage: 'radial-gradient(circle, #c1ff0005 1px, transparent 1px)',
          backgroundSize: '4px 4px'
        }}
      >
        <div className="px-6 py-4 border-b border-[#2A2E39]">
          <h2 className="text-lg font-medium text-(--color-neutral-100)">Agent M</h2>
        </div>
        
        <div className="p-6">
          <div className="mb-6">
            <h3 className="text-lg font-medium text-(--color-neutral-100) mb-6">Long Term View</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-6">
              {renderStatCards(ltvStats)}
            </div>
          </div>
        </div>
      </div>

      <div 
        className="rounded-lg bg-(--color-background) shadow-lg relative overflow-hidden border border-[#2A2E39]"
        style={{
          backgroundImage: 'radial-gradient(circle, #c1ff0005 1px, transparent 1px)',
          backgroundSize: '4px 4px'
        }}
      >
        <div className="px-6 py-4 border-b border-[#2A2E39]">
          <h2 className="text-lg font-medium text-(--color-neutral-100)">Ticker Pulse</h2>
        </div>
        
        <div className="p-6">
          <div className="mb-6">
            <h3 className="text-lg font-medium text-(--color-neutral-100) mb-6">Market Analysis</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-6">
              {renderStatCards(tickerPulseStats)}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AutomatedAnalytics; 