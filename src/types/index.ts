export type SourceType = 'sop' | 'email' | 'report' | 'structured';

export interface Source {
  source: string;
  source_type: SourceType;
  section: string;
  text: string;
  score: number;
}

export interface Plan {
  routes: string[];
  search_queries: string[];
  source_type_filter: string[] | null;
  reasoning: string;
  email_count_intent: boolean;
  thread_id: string | null;
}

export interface Claim {
  claim: string;
  supported: boolean;
}

export interface ChatResponse {
  answer: string;
  confidence: number;
  is_faithful: boolean;
  sources: Source[];
  plan: Plan;
  timeline: string[];
  claims: Claim[];
  errors: string[];
  latency_ms?: number;
}

export interface MessageMeta {
  confidence: number;
  sources: Source[];
  plan: Plan;
  timeline: string[];
  claims: Claim[];
  errors: string[];
  latency_ms?: number;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
  meta?: MessageMeta;
}

export interface Conversation {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: Message[];
}
