import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { Message } from '../types';
import { DetailPanel } from './DetailPanel';

interface MessageBubbleProps {
  message: Message;
  isStreaming?: boolean;
  streamedText?: string;
}

function AssistantAvatar() {
  return (
    <div className="assistant-avatar" aria-hidden="true">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
        <rect x="3" y="5" width="18" height="3" rx="1.5" fill="currentColor" opacity="0.9"/>
        <rect x="3" y="10.5" width="12" height="3" rx="1.5" fill="currentColor" opacity="0.9"/>
        <rect x="3" y="16" width="14" height="3" rx="1.5" fill="currentColor" opacity="0.9"/>
        <circle cx="20" cy="18" r="3.5" fill="#22c55e"/>
        <path d="M18.5 18l1 1 2-2" stroke="white" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round"/>
      </svg>
    </div>
  );
}

export function MessageBubble({ message, isStreaming, streamedText }: MessageBubbleProps) {
  const displayText = isStreaming && streamedText ? streamedText : message.content;

  if (message.role === 'user') {
    return (
      <div className="message message-user">
        <div className="user-bubble">
          <p>{message.content}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="message message-assistant">
      <AssistantAvatar />
      <div className="assistant-content">
        <div className="assistant-answer">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {displayText}
          </ReactMarkdown>
          {isStreaming && <span className="cursor-blink" aria-hidden="true" />}
        </div>
        {!isStreaming && message.meta && (
          <DetailPanel
            confidence={message.meta.confidence}
            sources={message.meta.sources}
            plan={message.meta.plan}
            timeline={message.meta.timeline}
            claims={message.meta.claims}
            errors={message.meta.errors}
            latency_ms={message.meta.latency_ms}
          />
        )}
      </div>
    </div>
  );
}
