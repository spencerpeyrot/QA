import { useState, useEffect } from 'react';
import { qaApi } from '../services/api';
import type { QAEvaluation } from '../services/api';
import { Trash2 } from 'lucide-react';
import AgentMetricsChart from './AgentMetricsChart';

export function Analytics() {
  const [evaluations, setEvaluations] = useState<QAEvaluation[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isEditing, setIsEditing] = useState(false);

  useEffect(() => {
    const fetchEvaluations = async () => {
      try {
        setIsLoading(true);
        setError(null);
        const data = await qaApi.listQAEvaluations();
        setEvaluations(data);
      } catch (err) {
        setError('Failed to load evaluations');
        console.error('Error fetching evaluations:', err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchEvaluations();
  }, []);

  const handleDelete = async (idToDelete: string) => {
    if (!window.confirm('Are you sure you want to delete this evaluation?')) {
      return;
    }
    
    try {
      await qaApi.deleteQAEvaluation(idToDelete);
      setEvaluations(currentEvaluations => 
        currentEvaluations.filter(evaluation => evaluation._id !== idToDelete)
      );
    } catch (err) {
      setError('Failed to delete evaluation');
      console.error('Error deleting evaluation:', err);
      setTimeout(() => setError(null), 3000);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="text-(--color-neutral-500)">Loading evaluations...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-md bg-red-500 bg-opacity-10 border border-red-500 p-4 text-red-500 mb-4">
        {error}
      </div>
    );
  }

  return (
    <div className="analytics-container flex flex-col gap-8">
      <div className="rounded-lg bg-(--color-background) p-6 shadow-lg relative overflow-hidden border border-[#2A2E39]">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-medium text-(--color-neutral-100)">QA Evaluations</h2>
          <button 
            onClick={() => setIsEditing(!isEditing)}
            className={`px-3 py-1 rounded-md text-sm font-medium border transition-colors
              ${isEditing 
                ? 'bg-(--color-accent) text-(--color-background) border-(--color-accent)'
                : 'bg-[#1a1d24] text-(--color-neutral-100) border-[#2A2E39] hover:border-(--color-accent)'
              }`}
          >
            {isEditing ? 'Done' : 'Edit'}
          </button>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-[#2A2E39]">
                <th className="text-left p-2 text-sm font-medium text-(--color-neutral-500)">Timestamp</th>
                <th className="text-left p-2 text-sm font-medium text-(--color-neutral-500)">Agent</th>
                <th className="text-left p-2 text-sm font-medium text-(--color-neutral-500)">Sub-component</th>
                <th className="text-center p-2 text-sm font-medium text-(--color-neutral-500)">QA Prompt Rating</th>
                <th className="text-center p-2 text-sm font-medium text-(--color-neutral-500)">Agent Output Rating</th>
                {isEditing && (
                  <th className="text-right p-2 text-sm font-medium text-(--color-neutral-500)">Actions</th>
                )}
              </tr>
            </thead>
            <tbody>
              {evaluations.map((evaluation) => (
                <tr key={evaluation._id} className="border-b border-[#2A2E39] hover:bg-[#1a1d24]">
                  <td className="p-2 text-sm text-(--color-neutral-100)">
                    {(() => {
                      const utcString = evaluation.created_at + 'Z';
                      const date = new Date(utcString);
                      return date.toLocaleString('en-US', {
                        year: 'numeric', month: 'numeric', day: 'numeric',
                        hour: 'numeric', minute: '2-digit', hour12: true,
                        timeZone: 'America/New_York'
                      });
                    })()}
                  </td>
                  <td className="p-2 text-sm text-(--color-neutral-100)">
                    {evaluation.agent}
                  </td>
                  <td className="p-2 text-sm text-(--color-neutral-100)">
                    {evaluation.sub_component || '-'}
                  </td>
                  <td className="p-2 text-sm text-(--color-neutral-100) text-center">
                    {evaluation.qa_rating === null ? '-' : evaluation.qa_rating ? 'Pass' : 'Fail'}
                  </td>
                  <td className="p-2 text-sm text-(--color-neutral-100) text-center">
                    {evaluation.report_rating ? `${evaluation.report_rating}/5` : '-'}
                  </td>
                  {isEditing && (
                    <td className="p-2 text-right">
                      <button 
                        onClick={() => handleDelete(evaluation._id)}
                        className="p-1 text-red-500 hover:text-red-400 hover:bg-red-500/10 rounded-md transition-colors"
                        title="Delete Evaluation"
                      >
                        <Trash2 size={16} />
                      </button>
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      
      <AgentMetricsChart />
    </div>
  );
} 