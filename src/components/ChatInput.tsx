import { useState, useRef, useEffect, type FormEvent, type KeyboardEvent } from 'react';

interface ChatInputProps {
  onSubmit: (text: string) => void;
  disabled: boolean;
}

export function ChatInput({ onSubmit, disabled }: ChatInputProps) {
  const [value, setValue] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (!disabled && textareaRef.current) {
      textareaRef.current.focus();
    }
  }, [disabled]);

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 200) + 'px';
  }, [value]);

  function handleSubmit(e?: FormEvent) {
    e?.preventDefault();
    if (!value.trim() || disabled) return;
    onSubmit(value.trim());
    setValue('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }

  return (
    <div className="input-bar-wrapper">
      <form className="input-bar" onSubmit={handleSubmit}>
        <textarea
          ref={textareaRef}
          className="input-textarea"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask anything about HCSA safety, permits, contractors…"
          disabled={disabled}
          rows={1}
          aria-label="Message input"
        />
        <button
          type="submit"
          className="send-btn"
          disabled={disabled || !value.trim()}
          aria-label="Send message"
        >
          {disabled ? (
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none" className="spin">
              <circle cx="9" cy="9" r="7" stroke="currentColor" strokeWidth="2" strokeDasharray="30 14" />
            </svg>
          ) : (
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
              <path d="M3 9h12M10 4l5 5-5 5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          )}
        </button>
      </form>
      <p className="input-hint">HCSA Assistant may make mistakes. Verify critical information against source documents.</p>
    </div>
  );
}
