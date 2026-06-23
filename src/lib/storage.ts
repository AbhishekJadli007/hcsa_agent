import { v4 as uuidv4 } from 'uuid';
import type { Conversation, Message, MessageMeta } from '../types';

const STORAGE_KEY = 'hcsa_conversations';
const ACTIVE_KEY = 'hcsa_active_conversation';

function load(): Conversation[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as Conversation[]) : [];
  } catch {
    return [];
  }
}

function save(conversations: Conversation[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
}

export function getConversations(): Conversation[] {
  return load().sort((a, b) => b.updatedAt - a.updatedAt);
}

export function getConversation(id: string): Conversation | undefined {
  return load().find((c) => c.id === id);
}

export function createConversation(): Conversation {
  const conv: Conversation = {
    id: uuidv4(),
    title: 'New conversation',
    createdAt: Date.now(),
    updatedAt: Date.now(),
    messages: [],
  };
  const all = load();
  all.push(conv);
  save(all);
  setActiveConversationId(conv.id);
  return conv;
}

export function deleteConversation(id: string): void {
  const all = load().filter((c) => c.id !== id);
  save(all);
  if (getActiveConversationId() === id) {
    localStorage.removeItem(ACTIVE_KEY);
  }
}

export function addUserMessage(conversationId: string, content: string): Message {
  const all = load();
  const conv = all.find((c) => c.id === conversationId);
  if (!conv) throw new Error('Conversation not found');

  const msg: Message = {
    id: uuidv4(),
    role: 'user',
    content,
    timestamp: Date.now(),
  };

  // Auto-title from first user message
  if (conv.messages.length === 0) {
    conv.title = content.slice(0, 60) + (content.length > 60 ? '…' : '');
  }

  conv.messages.push(msg);
  conv.updatedAt = Date.now();
  save(all);
  return msg;
}

export function addAssistantMessage(
  conversationId: string,
  content: string,
  meta: MessageMeta,
): Message {
  const all = load();
  const conv = all.find((c) => c.id === conversationId);
  if (!conv) throw new Error('Conversation not found');

  const msg: Message = {
    id: uuidv4(),
    role: 'assistant',
    content,
    timestamp: Date.now(),
    meta,
  };

  conv.messages.push(msg);
  conv.updatedAt = Date.now();
  save(all);
  return msg;
}

export function getActiveConversationId(): string | null {
  return localStorage.getItem(ACTIVE_KEY);
}

export function setActiveConversationId(id: string): void {
  localStorage.setItem(ACTIVE_KEY, id);
}
