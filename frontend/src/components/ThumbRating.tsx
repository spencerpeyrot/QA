import React from 'react';
import { ThumbsUp, ThumbsDown } from 'lucide-react';

interface ThumbRatingProps {
  value: boolean | null;
  onChange: (isGood: boolean) => void;
  label: string;
}

export function ThumbRating({ value, onChange, label }: ThumbRatingProps) {
  return (
    <div className="space-y-2">
      <p className="text-sm font-medium text-(--color-neutral-100)">
        {label}
      </p>
      <div className="flex gap-4">
        <button
          onClick={() => onChange(true)}
          className={`p-2 rounded-md transition-all duration-150 hover:bg-[#1a1d24] ${
            value === true ? 'text-green-500' : 'text-(--color-neutral-500)'
          }`}
        >
          <ThumbsUp size={24} />
        </button>
        <button
          onClick={() => onChange(false)}
          className={`p-2 rounded-md transition-all duration-150 hover:bg-[#1a1d24] ${
            value === false ? 'text-red-500' : 'text-(--color-neutral-500)'
          }`}
        >
          <ThumbsDown size={24} />
        </button>
      </div>
    </div>
  );
} 