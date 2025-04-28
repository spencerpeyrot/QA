import React from 'react';
import { StarRating } from './StarRating';
import { ThumbRating } from './ThumbRating';
import { ChevronRight, Clock, Hash, CheckCircle, RefreshCw } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import Skeleton from 'react-loading-skeleton';
import 'react-loading-skeleton/dist/skeleton.css';

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
  const isLoading = qaRun.response_markdown === 'Processing...';

  return (
    <div 
      className="rounded-lg bg-(--color-background) p-6 shadow-lg relative overflow-hidden border border-[#2A2E39]"
      style={{
        backgroundImage: 'radial-gradient(circle, #c1ff0005 1px, transparent 1px)',
        backgroundSize: '4px 4px'
      }}
    >
      {/* Header with metadata */}
      <div className="flex items-center justify-between border-b border-[#2A2E39] pb-4 mb-4">
        <div className="space-y-1">
          <h2 className="text-lg font-medium text-(--color-neutral-100)">
            QA Evaluation
          </h2>
          <div className="flex items-center gap-1 text-sm text-(--color-neutral-500)">
            <Clock size={14} className="stroke-(--color-neutral-500)" />
            <span>
              {isLoading ? (
                <Skeleton width={120} baseColor="#2A2E39" highlightColor="#3F4451" />
              ) : (
                (() => {
                  const utcString = qaRun.created_at + 'Z';
                  const date = new Date(utcString);
                  return date.toLocaleString('en-US', {
                    year: 'numeric', month: 'numeric', day: 'numeric',
                    hour: 'numeric', minute: '2-digit', hour12: true,
                    timeZone: 'America/New_York'
                  });
                })()
              )}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-1 text-sm text-(--color-neutral-500)">
          <Hash size={14} className="stroke-(--color-neutral-500)" />
          <span>
            {isLoading ? (
              <Skeleton width={60} baseColor="#2A2E39" highlightColor="#3F4451" />
            ) : (
              qaRun.id
            )}
          </span>
        </div>
      </div>

      {/* Agent and Component Info */}
      <div className="flex items-center gap-4 text-sm py-4 mb-4 border-t border-[#2A2E39]">
        <span className="text-(--color-neutral-100)">
          Agent: {isLoading ? (
            <Skeleton width={40} baseColor="#2A2E39" highlightColor="#3F4451" />
          ) : (
            <span className="text-(--color-accent)">{qaRun.agent}</span>
          )}
        </span>
        {(isLoading || qaRun.sub_component) && (
          <>
            <ChevronRight size={16} className="stroke-(--color-neutral-500)" />
            <span className="text-(--color-neutral-100)">
              {isLoading ? (
                <Skeleton width={100} baseColor="#2A2E39" highlightColor="#3F4451" />
              ) : (
                qaRun.sub_component
              )}
            </span>
          </>
        )}
      </div>

      {/* LLM Response */}
      <div className="rounded-md bg-(--color-background) p-4 border border-[#2A2E39] mb-4">
        {isLoading ? (
          <div className="space-y-4">
            <Skeleton count={3} height={20} baseColor="#2A2E39" highlightColor="#3F4451" />
            <Skeleton count={2} height={20} baseColor="#2A2E39" highlightColor="#3F4451" />
            <Skeleton count={4} height={20} baseColor="#2A2E39" highlightColor="#3F4451" />
            <Skeleton count={2} height={20} baseColor="#2A2E39" highlightColor="#3F4451" />
          </div>
        ) : (
          <div className="prose prose-invert max-w-none">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {qaRun.response_markdown}
            </ReactMarkdown>
          </div>
        )}
      </div>

      {/* Rating Section */}
      <div className="pt-4">
        <div className="flex items-start justify-between gap-6">
          <div className="flex-1">
            {/* QA Quality Rating */}
            {isLoading ? (
              <div className="space-y-2">
                <Skeleton width={120} height={16} baseColor="#2A2E39" highlightColor="#3F4451" />
                <Skeleton width={200} height={32} baseColor="#2A2E39" highlightColor="#3F4451" />
              </div>
            ) : (
              <ThumbRating
                value={qaRun.qa_rating}
                onChange={(isGood) => onQARating(qaRun.id, isGood)}
                label="QA Prompt Quality"
              />
            )}
          </div>
          <div className="flex-1">
            {/* Report Accuracy Rating - Only show if QA is good */}
            {qaRun.qa_rating === true && !isLoading && (
              <StarRating
                value={qaRun.report_rating}
                onChange={(rating) => onReportRating(qaRun.id, rating)}
                label="Agent Output Accuracy"
              />
            )}
            {isLoading && (
              <div className="space-y-2">
                <Skeleton width={120} height={16} baseColor="#2A2E39" highlightColor="#3F4451" />
                <Skeleton width={200} height={32} baseColor="#2A2E39" highlightColor="#3F4451" />
              </div>
            )}
          </div>
          <div className="flex items-center pt-7">
            {isLoading ? (
              <Skeleton width={100} height={32} baseColor="#2A2E39" highlightColor="#3F4451" />
            ) : showRetry ? (
              <button
                onClick={onRetry}
                className="flex items-center gap-1.5 py-1.5 px-3 rounded-md bg-(--color-accent) hover:bg-opacity-90 text-(--color-neutral-900) text-sm font-medium"
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