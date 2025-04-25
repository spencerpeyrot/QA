import { useState, useEffect } from 'react';
import { qaApi } from '../services/api';
import type { QAEvaluation } from '../services/api';

export function Analytics() {
  const [evaluations, setEvaluations] = useState<QAEvaluation[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchEvaluations = async () => {
      try {
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

  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="text-(--color-neutral-500)">Loading evaluations...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-md bg-red-500 bg-opacity-10 border border-red-500 p-4 text-red-500">
        {error}
      </div>
    );
  }

  return (
    <div className="rounded-lg bg-(--color-background) p-6 shadow-lg relative overflow-hidden border border-[#2A2E39]">
      <h2 className="text-lg font-medium text-(--color-neutral-100) mb-4">QA Evaluations</h2>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-[#2A2E39]">
              <th className="text-left p-2 text-sm font-medium text-(--color-neutral-500)">Date</th>
              <th className="text-left p-2 text-sm font-medium text-(--color-neutral-500)">Agent</th>
              <th className="text-left p-2 text-sm font-medium text-(--color-neutral-500)">Sub-component</th>
              <th className="text-left p-2 text-sm font-medium text-(--color-neutral-500)">QA Rating</th>
              <th className="text-left p-2 text-sm font-medium text-(--color-neutral-500)">Report Rating</th>
            </tr>
          </thead>
          <tbody>
            {evaluations.map((evaluation) => (
              <tr key={evaluation._id} className="border-b border-[#2A2E39] hover:bg-[#1a1d24]">
                <td className="p-2 text-sm text-(--color-neutral-100)">
                  {new Date(evaluation.created_at).toLocaleString()}
                </td>
                <td className="p-2 text-sm text-(--color-neutral-100)">
                  {evaluation.agent}
                </td>
                <td className="p-2 text-sm text-(--color-neutral-100)">
                  {evaluation.sub_component || '-'}
                </td>
                <td className="p-2 text-sm text-(--color-neutral-100)">
                  {evaluation.qa_rating ? `${evaluation.qa_rating}/5` : '-'}
                </td>
                <td className="p-2 text-sm text-(--color-neutral-100)">
                  {evaluation.report_rating ? `${evaluation.report_rating}/5` : '-'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
} 