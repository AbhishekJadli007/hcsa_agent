import type { SourceType } from '../types';

const CONFIG: Record<SourceType, { label: string; bg: string; text: string; border: string }> = {
  sop: {
    label: 'SOP / Policy',
    bg: 'var(--badge-sop-bg)',
    text: 'var(--badge-sop-text)',
    border: 'var(--badge-sop-border)',
  },
  email: {
    label: 'Email',
    bg: 'var(--badge-email-bg)',
    text: 'var(--badge-email-text)',
    border: 'var(--badge-email-border)',
  },
  report: {
    label: 'Annual Report',
    bg: 'var(--badge-report-bg)',
    text: 'var(--badge-report-text)',
    border: 'var(--badge-report-border)',
  },
  structured: {
    label: 'Structured Data',
    bg: 'var(--badge-struct-bg)',
    text: 'var(--badge-struct-text)',
    border: 'var(--badge-struct-border)',
  },
};

interface SourceTagProps {
  type: SourceType;
}

export function SourceTag({ type }: SourceTagProps) {
  const c = CONFIG[type] ?? CONFIG.sop;
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        padding: '2px 8px',
        borderRadius: '999px',
        fontSize: '0.7rem',
        fontWeight: 600,
        letterSpacing: '0.03em',
        textTransform: 'uppercase',
        background: c.bg,
        color: c.text,
        border: `1px solid ${c.border}`,
        whiteSpace: 'nowrap',
      }}
    >
      {c.label}
    </span>
  );
}
