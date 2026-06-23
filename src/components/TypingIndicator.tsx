interface TypingIndicatorProps {
  statusLine?: string;
}

export function TypingIndicator({ statusLine }: TypingIndicatorProps) {
  return (
    <div className="message message-assistant">
      <div className="assistant-avatar" aria-hidden="true">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
          <rect x="3" y="5" width="18" height="3" rx="1.5" fill="currentColor" opacity="0.9"/>
          <rect x="3" y="10.5" width="12" height="3" rx="1.5" fill="currentColor" opacity="0.9"/>
          <rect x="3" y="16" width="14" height="3" rx="1.5" fill="currentColor" opacity="0.9"/>
          <circle cx="20" cy="18" r="3.5" fill="#22c55e"/>
        </svg>
      </div>
      <div className="assistant-content">
        {statusLine ? (
          <div className="status-line">
            <span className="status-dot" />
            <span>{statusLine}</span>
          </div>
        ) : (
          <div className="typing-dots" aria-label="Loading response">
            <span />
            <span />
            <span />
          </div>
        )}
      </div>
    </div>
  );
}
