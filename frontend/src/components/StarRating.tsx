import React, { useState } from 'react';
import { Star } from 'lucide-react';

interface StarRatingProps {
  value: number;
  onChange: (rating: number) => void;
  label: string;
}

export function StarRating({ value, onChange, label }: StarRatingProps) {
  const [hoverValue, setHoverValue] = useState<number>(0);

  const getStarProps = (starPosition: number) => {
    const rating = hoverValue || value;
    const isFilled = starPosition <= rating;
    
    return {
      size: 24,
      strokeWidth: 1.5,
      className: `transition-all duration-150 hover:scale-110 cursor-pointer ${
        isFilled 
          ? 'fill-(--color-accent) stroke-(--color-accent)' 
          : 'fill-transparent stroke-(--color-neutral-500) hover:stroke-(--color-accent)'
      }`,
    };
  };

  return (
    <div className="space-y-2">
      <p className="text-sm font-medium text-(--color-neutral-100)">
        {label} ({value || '0'}/5)
      </p>
      <div className="flex gap-2">
        {[1, 2, 3, 4, 5].map((star) => (
          <button
            key={star}
            onClick={() => onChange(star)}
            onMouseEnter={() => setHoverValue(star)}
            onMouseLeave={() => setHoverValue(0)}
            className="focus:outline-none"
          >
            <Star {...getStarProps(star)} />
          </button>
        ))}
      </div>
    </div>
  );
} 