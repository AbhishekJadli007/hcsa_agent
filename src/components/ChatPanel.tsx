import { useRef, useEffect } from 'react';
import type { Message } from '../types';
import { MessageBubble } from './MessageBubble';
import { TypingIndicator } from './TypingIndicator';
import { EmptyState } from './EmptyState';
import { ChatInput } from './ChatInput';

interface ChatPanelProps {
  messages: Message[];
  isLoading: boolean;
  streamedText: string;
  statusLine: string;
  onSubmit: (text: string) => void;
}

export function ChatPanel({ messages, isLoading, streamedText, statusLine, onSubmit }: ChatPanelProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamedText, isLoading]);

  const showStreaming = isLoading && streamedText;

  return (
    <div className="chat-panel">
      <div className="chat-scroll">
        <div className="chat-content">
          {messages.length === 0 && !isLoading ? (
            <EmptyState onSubmit={onSubmit} />
          ) : (
            <>
              {messages.map((msg, i) => {
                const isLastAssistant =
                  msg.role === 'assistant' && i === messages.length - 1;
                return (
                  <MessageBubble
                    key={msg.id}
                    message={msg}
                    isStreaming={isLastAssistant && showStreaming ? true : false}
                    streamedText={isLastAssistant && showStreaming ? streamedText : undefined}
                  />
                );
              })}
              {isLoading && !streamedText && (
                <TypingIndicator statusLine={statusLine} />
              )}
            </>
          )}
          <div ref={bottomRef} />
        </div>
      </div>
      <ChatInput onSubmit={onSubmit} disabled={isLoading} />
    </div>
  );
}
