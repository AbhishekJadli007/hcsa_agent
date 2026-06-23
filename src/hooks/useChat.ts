import { useState, useEffect, useCallback } from 'react';
import type { Conversation } from '../types';
import {
  getConversations,
  createConversation,
  deleteConversation,
  addUserMessage,
  addAssistantMessage,
  getActiveConversationId,
  setActiveConversationId,
} from '../lib/storage';
import { sendMessage } from '../lib/api';

export function useChat() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [streamedText, setStreamedText] = useState('');
  const [statusLine, setStatusLine] = useState('');

  const refresh = useCallback(() => {
    setConversations(getConversations());
  }, []);

  useEffect(() => {
    refresh();
    const savedId = getActiveConversationId();
    const all = getConversations();
    if (savedId && all.find((c) => c.id === savedId)) {
      setActiveId(savedId);
    } else if (all.length > 0) {
      setActiveId(all[0].id);
      setActiveConversationId(all[0].id);
    }
  }, [refresh]);

  const activeConversation = activeId
    ? conversations.find((c) => c.id === activeId) ?? null
    : null;

  const startNewChat = useCallback(() => {
    const conv = createConversation();
    setActiveId(conv.id);
    refresh();
  }, [refresh]);

  const selectConversation = useCallback(
    (id: string) => {
      setActiveConversationId(id);
      setActiveId(id);
    },
    [],
  );

  const removeConversation = useCallback(
    (id: string) => {
      deleteConversation(id);
      const remaining = getConversations();
      if (activeId === id) {
        if (remaining.length > 0) {
          setActiveId(remaining[0].id);
          setActiveConversationId(remaining[0].id);
        } else {
          setActiveId(null);
        }
      }
      refresh();
    },
    [activeId, refresh],
  );

  const submitMessage = useCallback(
    async (text: string) => {
      if (!text.trim() || isLoading) return;

      let convId = activeId;
      if (!convId) {
        const conv = createConversation();
        convId = conv.id;
        setActiveId(conv.id);
      }

      addUserMessage(convId, text.trim());
      refresh();
      setIsLoading(true);
      setStreamedText('');
      setStatusLine('Routing query across agents…');

      try {
        const response = await sendMessage(text.trim());

        // Typewriter effect
        const fullText = response.answer;
        const words = fullText.split(' ');
        setStatusLine('');

        let built = '';
        for (let i = 0; i < words.length; i++) {
          built += (i === 0 ? '' : ' ') + words[i];
          setStreamedText(built);
          if (i % 4 === 0) {
            await new Promise((r) => setTimeout(r, 18));
          }
        }

        addAssistantMessage(convId, fullText, {
          confidence: response.confidence,
          sources: response.sources,
          plan: response.plan,
          timeline: response.timeline,
          claims: response.claims,
          errors: response.errors,
          latency_ms: response.latency_ms,
        });

        setStreamedText('');
        refresh();
      } catch (err) {
        const errMsg = err instanceof Error ? err.message : 'An unexpected error occurred.';
        addAssistantMessage(convId, `**Error:** ${errMsg}`, {
          confidence: 0,
          sources: [],
          plan: { routes: [], search_queries: [], source_type_filter: null, reasoning: '', email_count_intent: false, thread_id: null },
          timeline: [],
          claims: [],
          errors: [errMsg],
        });
        setStreamedText('');
        refresh();
      } finally {
        setIsLoading(false);
        setStatusLine('');
      }
    },
    [activeId, isLoading, refresh],
  );

  return {
    conversations,
    activeConversation,
    activeId,
    isLoading,
    streamedText,
    statusLine,
    startNewChat,
    selectConversation,
    removeConversation,
    submitMessage,
  };
}
