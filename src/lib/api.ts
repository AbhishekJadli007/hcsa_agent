import type { ChatResponse } from '../types';

export const USE_MOCK = false;

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';

const MOCK_RESPONSE: ChatResponse = {
  answer: `## Working at Height — Permit Requirements

Based on **SOP-CO-003 Section 4.2**, any work involving a fall risk exceeding **2 metres** requires a **Working at Height (WAH) Permit** before work commences.

### Required Permits

| Permit Type | Trigger Condition | Valid For |
|---|---|---|
| Working at Height (WAH) | Height ≥ 2 m | Up to 7 calendar days |
| Scaffold Erection Permit | Scaffold > 3 m high | Duration of erection |
| Hot Work Permit | Welding / grinding near WAH | Per session, max 8 hrs |

### Application Process

1. **Site supervisor** completes Form PTW-01 at least **24 hours** before work.
2. **Safety Officer** inspects the work area and countersigns.
3. **Project Manager** gives final approval for heights ≥ 10 m.
4. Permit is displayed at the **workface** throughout the operation.

> **Note:** Permits cannot be verbally extended. A new PTW-01 must be submitted if work extends beyond the approved validity period. [SOP-CO-003 Section 4.2.7]`,
  confidence: 0.91,
  is_faithful: true,
  sources: [
    {
      source: 'SOP-CO-003.pdf',
      source_type: 'sop',
      section: '4.2 Permit to Work Requirements',
      text: 'Any work activity involving a fall risk exceeding 2 metres above ground level requires a valid Working at Height Permit prior to commencement.',
      score: 9.12,
    },
    {
      source: 'SOP-CO-003.pdf',
      source_type: 'sop',
      section: '4.2.7 Permit Validity and Extensions',
      text: 'Permits are valid for the duration specified, up to a maximum of 7 calendar days. Extensions require a new application; verbal extensions are not permitted.',
      score: 7.54,
    },
    {
      source: 'Email_12.pdf',
      source_type: 'email',
      section: 'Email 2 of 3',
      text: 'Please be reminded that all WAH permits for the Tanjung Pagar site must be submitted to the safety office by 0800 the preceding day.',
      score: 4.21,
    },
  ],
  plan: {
    routes: ['vector_agent'],
    search_queries: [
      'working at height permit requirements WAH PTW process',
      'permit to work application approval scaffold hot work',
      'WAH permit validity extension SOP-CO-003',
    ],
    source_type_filter: ['sop'],
    reasoning: 'Policy and SOP documents hold permit requirements; vector search is appropriate.',
  },
  timeline: [
    'Planner -> routes: ["vector_agent"] | 3 sub-queries | source_filter: ["sop"]',
    'VectorAgent: multi-query (3 sub-queries) dense+BM25+RRF+rerank → 8 chunks',
    'Verifier: faithfulness 91% (10/11 claims supported, batched)',
  ],
  claims: [
    { claim: 'A Working at Height Permit is required for fall risks exceeding 2 metres.', supported: true },
    { claim: 'WAH permits are valid for up to 7 calendar days.', supported: true },
    { claim: 'A Scaffold Erection Permit is required for scaffolds over 3 m high.', supported: true },
    { claim: 'Hot Work Permits are valid per session with a maximum of 8 hours.', supported: true },
    { claim: 'Form PTW-01 must be submitted at least 24 hours before work.', supported: true },
    { claim: 'Project Manager approval is required for heights ≥ 10 m.', supported: false },
    { claim: 'Permits must be displayed at the workface throughout the operation.', supported: true },
  ],
  errors: [],
};

async function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

export async function sendMessage(message: string): Promise<ChatResponse> {
  if (USE_MOCK) {
    await sleep(1200);
    return MOCK_RESPONSE;
  }

  const res = await fetch(`${API_BASE}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`API error ${res.status}: ${text || res.statusText}`);
  }

  return res.json() as Promise<ChatResponse>;
}

const HARDCODED_EXAMPLES = [
  'What permits are required for working at height on a scaffold?',
  'A worker fell 2.5 m from a scaffold and sprained their ankle. What is the incident severity level and what are the reporting timelines?',
  "What were HDB's Key Audit Matters for FY 2022/23 and FY 2023/24?",
  'What is the performance summary for contractor CONTR-2022-047?',
];

export async function getExamples(): Promise<string[]> {
  if (USE_MOCK) return HARDCODED_EXAMPLES;
  try {
    const res = await fetch(`${API_BASE}/api/examples`);
    if (!res.ok) return HARDCODED_EXAMPLES;
    const data = await res.json();
    return data.examples ?? HARDCODED_EXAMPLES;
  } catch {
    return HARDCODED_EXAMPLES;
  }
}
