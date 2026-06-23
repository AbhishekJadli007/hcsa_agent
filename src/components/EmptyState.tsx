import { useState, useEffect } from 'react';
import { getExamples } from '../lib/api';

const DEFAULT_EXAMPLES = [
  'What permits are required for working at height on a scaffold?',
  'A worker fell 2.5 m and sprained their ankle. What is the incident severity and reporting timeline?',
  "What were HDB's Key Audit Matters for FY 2022/23 and FY 2023/24?",
  'Give me the full performance summary for contractor CONTR-2022-047.',
];

interface EmptyStateProps {
  onSubmit: (q: string) => void;
}

export function EmptyState({ onSubmit }: EmptyStateProps) {
  const [examples, setExamples] = useState<string[]>(DEFAULT_EXAMPLES);

  useEffect(() => {
    getExamples().then(setExamples).catch(() => {});
  }, []);

  return (
    <div className="empty-state">
      <div className="empty-logo" aria-hidden="true">
        <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
          <rect width="48" height="48" rx="12" fill="var(--primary)"/>
          <rect x="10" y="14" width="28" height="5" rx="2.5" fill="white" opacity="0.9"/>
          <rect x="10" y="22" width="20" height="5" rx="2.5" fill="white" opacity="0.9"/>
          <rect x="10" y="30" width="22" height="5" rx="2.5" fill="white" opacity="0.9"/>
          <circle cx="38" cy="34" r="7" fill="#22c55e"/>
          <path d="M35 34l2 2 4-4" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </div>
      <h2 className="empty-title">HCSA Intelligent Assistant</h2>
      <p className="empty-subtitle">
        Ask about safety procedures, permits, incident reporting, contractor performance,
        annual reports, and more.
      </p>

      <div className="examples-grid">
        {examples.map((ex, i) => (
          <button
            key={i}
            className="example-card"
            onClick={() => onSubmit(ex)}
          >
            <span className="example-icon" aria-hidden="true">
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path d="M8 2a6 6 0 100 12A6 6 0 008 2zm0 10.5a4.5 4.5 0 110-9 4.5 4.5 0 010 9zm-.75-6.75V8.5l2 1.25.5-.75-1.75-1V5.75H7.25z" fill="currentColor"/>
              </svg>
            </span>
            <span>{ex}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
