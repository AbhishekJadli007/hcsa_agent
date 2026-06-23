interface ConfidenceBadgeProps {
  score: number;
}

export function ConfidenceBadge({ score }: ConfidenceBadgeProps) {
  const pct = Math.round(score * 100);

  let label: string;
  let cls: string;
  if (score >= 0.8) {
    label = 'High confidence';
    cls = 'confidence-high';
  } else if (score >= 0.5) {
    label = 'Medium confidence';
    cls = 'confidence-medium';
  } else {
    label = 'Low — verify against sources';
    cls = 'confidence-low';
  }

  return (
    <div className={`confidence-badge ${cls}`}>
      <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
        <circle cx="6" cy="6" r="5" stroke="currentColor" strokeWidth="1.5" />
        <path d="M4 6l1.5 1.5L8.5 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className={score < 0.5 ? 'hidden' : ''} />
        {score < 0.5 && (
          <path d="M4 4l4 4M8 4l-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        )}
      </svg>
      <span className="confidence-pct">{pct}%</span>
      <span className="confidence-label">{label}</span>
    </div>
  );
}
