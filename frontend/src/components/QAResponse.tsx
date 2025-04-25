import React from 'react';
import { StarRating } from './StarRating';
import { ThumbRating } from './ThumbRating';
import { ChevronRight, Clock, Hash, CheckCircle, RefreshCw } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface QAResponseProps {
  qaRun: {
    id: string;
    agent: string;
    sub_component: string | null;
    report_text: string;
    variables: Record<string, string>;
    response_markdown: string;
    qa_rating: boolean | null;
    report_rating: number;
    created_at: string;
  };
  onQARating: (id: string, isGood: boolean) => void;
  onReportRating: (id: string, rating: number) => void;
  onSubmit?: () => void;
  onRetry?: () => void;
}

export function QAResponse({ qaRun, onQARating, onReportRating, onSubmit, onRetry }: QAResponseProps) {
  const showRetry = qaRun.qa_rating === false;
  const canSubmit = qaRun.qa_rating === true && qaRun.report_rating > 0;

  return (
    <div className="space-y-6 rounded-lg bg-(--color-background) p-6 shadow-lg relative overflow-hidden border border-[#2A2E39]">
      {/* Header with metadata */}
      <div className="flex items-center justify-between border-b border-[#2A2E39] pb-4">
        <div className="space-y-1">
          <h3 className="text-lg font-medium text-(--color-neutral-100)">
            QA Evaluation
          </h3>
          <div className="flex items-center gap-1 text-sm text-(--color-neutral-500)">
            <Clock size={14} className="stroke-(--color-neutral-500)" />
            <span>
              {(() => {
                const utcString = qaRun.created_at + 'Z';
                const date = new Date(utcString);
                return date.toLocaleString('en-US', {
                  year: 'numeric', month: 'numeric', day: 'numeric',
                  hour: 'numeric', minute: '2-digit', hour12: true,
                  timeZone: 'America/New_York'
                });
              })()}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-1 text-sm text-(--color-neutral-500)">
          <Hash size={14} className="stroke-(--color-neutral-500)" />
          <span>{qaRun.id}</span>
        </div>
      </div>

      {/* Agent and Component Info */}
      <div className="flex items-center gap-4 text-sm">
        <span className="text-(--color-neutral-100)">
          Agent: <span className="text-(--color-accent)">{qaRun.agent}</span>
        </span>
        {qaRun.sub_component && (
          <>
            <ChevronRight size={16} className="stroke-(--color-neutral-500)" />
            <span className="text-(--color-neutral-100)">
              {qaRun.sub_component}
            </span>
          </>
        )}
      </div>

      {/* LLM Response */}
      <div className="rounded-md bg-[#1a1d24] p-4 border border-[#2A2E39]">
        <div className="prose prose-invert max-w-none">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {qaRun.response_markdown}
          </ReactMarkdown>
        </div>
      </div>

      {/* Rating Section */}
      <div className="pt-4 border-t border-[#2A2E39]">
        <div className="flex items-start justify-between gap-6">
          <div className="flex-1">
            {/* QA Quality Rating */}
            <ThumbRating
              value={qaRun.qa_rating}
              onChange={(isGood) => onQARating(qaRun.id, isGood)}
              label="QA Prompt Quality"
            />
          </div>
          <div className="flex-1">
            {/* Report Accuracy Rating - Only show if QA is good */}
            {qaRun.qa_rating === true && (
              <StarRating
                value={qaRun.report_rating}
                onChange={(rating) => onReportRating(qaRun.id, rating)}
                label="Agent Output Accuracy"
              />
            )}
          </div>
          <div className="flex items-center pt-7">
            {showRetry ? (
              <button
                onClick={onRetry}
                className="flex items-center gap-1.5 py-1.5 px-3 rounded-md bg-yellow-500 hover:bg-yellow-600 text-black text-sm font-medium"
              >
                <RefreshCw size={16} />
                <span>Retry QA</span>
              </button>
            ) : (
              <button
                onClick={onSubmit}
                disabled={!canSubmit}
                className={`flex items-center gap-1.5 py-1.5 px-3 rounded-md transition-all duration-150 text-sm
                  ${canSubmit 
                    ? 'bg-(--color-accent) text-(--color-background) hover:opacity-90' 
                    : 'bg-(--color-neutral-800) text-(--color-neutral-500) cursor-not-allowed'
                  }`}
              >
                <CheckCircle size={16} />
                <span>Submit</span>
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
} 