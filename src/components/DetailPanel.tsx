import { useState } from 'react';
import type { Source, Plan, Claim } from '../types';
import { SourceTag } from './SourceTag';
import { ConfidenceBadge } from './ConfidenceBadge';

interface DetailPanelProps {
  confidence: number;
  sources: Source[];
  plan: Plan;
  timeline: string[];
  claims: Claim[];
  errors: string[];
  latency_ms?: number;
}

type Tab = 'sources' | 'analysis';

export function DetailPanel({
  confidence,
  sources,
  plan,
  timeline,
  claims,
  errors,
  latency_ms,
}: DetailPanelProps) {
  const [open, setOpen] = useState(false);
  const [tab, setTab] = useState<Tab>('sources');

  const sorted = [...sources].sort((a, b) => b.score - a.score);

  return (
    <div className="detail-panel">
      <div className="detail-panel-header">
        <ConfidenceBadge score={confidence} />
        <div className="detail-panel-header-right">
          {latency_ms !== undefined && (
            <span className="latency-badge" title="Server-side pipeline latency">
              {latency_ms >= 1000
                ? `${(latency_ms / 1000).toFixed(1)}s`
                : `${latency_ms}ms`}
            </span>
          )}
          <button
            className="detail-toggle"
            onClick={() => setOpen((v) => !v)}
            aria-expanded={open}
          >
            <span>{open ? 'Hide details' : 'Show details'}</span>
            <svg
              width="14"
              height="14"
              viewBox="0 0 14 14"
              fill="none"
              style={{ transform: open ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s' }}
            >
              <path d="M2 5l5 5 5-5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        </div>
      </div>

      {errors.length > 0 && (
        <div className="error-banner">
          <strong>Pipeline errors:</strong>
          <ul>
            {errors.map((e, i) => (
              <li key={i}>{e}</li>
            ))}
          </ul>
        </div>
      )}

      {open && (
        <div className="detail-body">
          <div className="tab-bar">
            <button
              className={`tab-btn ${tab === 'sources' ? 'active' : ''}`}
              onClick={() => setTab('sources')}
            >
              Sources
              {sorted.length > 0 && <span className="tab-count">{sorted.length}</span>}
            </button>
            <button
              className={`tab-btn ${tab === 'analysis' ? 'active' : ''}`}
              onClick={() => setTab('analysis')}
            >
              Analysis
            </button>
          </div>

          {tab === 'sources' && (
            <div className="sources-list">
              {sorted.length === 0 ? (
                <p className="empty-hint">No sources retrieved for this response.</p>
              ) : (
                sorted.map((s, i) => (
                  <div className="source-card" key={i}>
                    <div className="source-card-header">
                      <span className="source-filename">{s.source}</span>
                      <SourceTag type={s.source_type} />
                      <span className="source-score">score {s.score.toFixed(2)}</span>
                    </div>
                    {s.section && (
                      <div className="source-section">{s.section}</div>
                    )}
                    <p className="source-text">{s.text}</p>
                  </div>
                ))
              )}
            </div>
          )}

          {tab === 'analysis' && (
            <div className="analysis-body">
              {/* Routing plan */}
              <div className="analysis-section">
                <h4 className="analysis-heading">Query Router Plan</h4>
                <div className="plan-routes">
                  {plan.routes.map((r) => (
                    <span className="route-pill" key={r}>{r.replace('_agent', '')}</span>
                  ))}
                  {plan.email_count_intent && (
                    <span className="route-pill route-pill-email">email count</span>
                  )}
                </div>
                {plan.reasoning && (
                  <p className="plan-reasoning">{plan.reasoning}</p>
                )}
                {plan.thread_id && (
                  <div className="plan-filter">
                    Thread: <code>{plan.thread_id}</code>
                  </div>
                )}
                {plan.search_queries.length > 0 && (
                  <>
                    <div className="analysis-sublabel">Sub-queries ({plan.search_queries.length})</div>
                    <ol className="subquery-list">
                      {plan.search_queries.map((q, i) => (
                        <li key={i}>{q}</li>
                      ))}
                    </ol>
                  </>
                )}
                {plan.source_type_filter && plan.source_type_filter.length > 0 && (
                  <div className="plan-filter">
                    Source filter:{' '}
                    {plan.source_type_filter.map((f) => (
                      <code key={f}>{f}</code>
                    ))}
                  </div>
                )}
              </div>

              {/* Execution timeline */}
              {timeline.length > 0 && (
                <div className="analysis-section">
                  <h4 className="analysis-heading">Execution Timeline</h4>
                  <ol className="timeline-list">
                    {timeline.map((step, i) => (
                      <li key={i}>{step}</li>
                    ))}
                  </ol>
                </div>
              )}

              {/* Claims */}
              {claims.length > 0 && (
                <div className="analysis-section">
                  <h4 className="analysis-heading">
                    Claim Faithfulness
                    <span className="claims-ratio">
                      {claims.filter((c) => c.supported).length}/{claims.length} supported
                    </span>
                  </h4>
                  <ul className="claims-list">
                    {claims.map((c, i) => (
                      <li key={i} className={`claim-item ${c.supported ? 'supported' : 'unsupported'}`}>
                        <span className="claim-icon" aria-hidden="true">
                          {c.supported ? '✓' : '✗'}
                        </span>
                        <span>{c.claim}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
