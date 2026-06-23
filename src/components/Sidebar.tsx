import type { Conversation } from '../types';

interface SidebarProps {
  conversations: Conversation[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  onToggle: () => void;
  darkMode: boolean;
  onToggleDark: () => void;
}

function formatDate(ts: number): string {
  const d = new Date(ts);
  const now = new Date();
  const diff = now.getTime() - d.getTime();
  if (diff < 86400000) return 'Today';
  if (diff < 172800000) return 'Yesterday';
  return d.toLocaleDateString('en-SG', { day: 'numeric', month: 'short' });
}

function groupByDate(convs: Conversation[]) {
  const groups: { label: string; items: Conversation[] }[] = [];
  const seen = new Map<string, number>();
  for (const c of convs) {
    const label = formatDate(c.updatedAt);
    if (!seen.has(label)) {
      seen.set(label, groups.length);
      groups.push({ label, items: [] });
    }
    groups[seen.get(label)!].items.push(c);
  }
  return groups;
}

export function Sidebar({
  conversations,
  activeId,
  onSelect,
  onNew,
  onDelete,
  onToggle,
  darkMode,
  onToggleDark,
}: SidebarProps) {
  const groups = groupByDate(conversations);

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-brand">
          <svg width="28" height="28" viewBox="0 0 28 28" fill="none" aria-hidden="true">
            <rect width="28" height="28" rx="7" fill="var(--primary)"/>
            <rect x="6" y="8" width="16" height="2.5" rx="1.25" fill="white" opacity="0.9"/>
            <rect x="6" y="12.5" width="11" height="2.5" rx="1.25" fill="white" opacity="0.9"/>
            <rect x="6" y="17" width="13" height="2.5" rx="1.25" fill="white" opacity="0.9"/>
            <circle cx="22" cy="20" r="4" fill="#22c55e"/>
            <path d="M20.5 20l1 1 2-2" stroke="white" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          <span>HCSA Assistant</span>
        </div>
        <button className="sidebar-close" onClick={onToggle} aria-label="Close sidebar">
          <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
            <path d="M3 9h12M3 4h12M3 14h12" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"/>
          </svg>
        </button>
      </div>

      <button className="new-chat-btn" onClick={onNew}>
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
          <path d="M8 2v12M2 8h12" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
        </svg>
        New chat
      </button>

      <nav className="sidebar-nav">
        {conversations.length === 0 ? (
          <p className="sidebar-empty">No conversations yet.</p>
        ) : (
          groups.map((group) => (
            <div key={group.label} className="sidebar-group">
              <div className="sidebar-group-label">{group.label}</div>
              {group.items.map((conv) => (
                <div
                  key={conv.id}
                  className={`sidebar-item ${conv.id === activeId ? 'active' : ''}`}
                >
                  <button
                    className="sidebar-item-btn"
                    onClick={() => onSelect(conv.id)}
                    title={conv.title}
                  >
                    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" className="chat-icon" aria-hidden="true">
                      <path d="M2 2h10a1 1 0 011 1v6a1 1 0 01-1 1H5l-3 2V3a1 1 0 011-1z" stroke="currentColor" strokeWidth="1.2" fill="none"/>
                    </svg>
                    <span className="sidebar-item-title">{conv.title}</span>
                  </button>
                  <button
                    className="sidebar-delete-btn"
                    onClick={(e) => {
                      e.stopPropagation();
                      onDelete(conv.id);
                    }}
                    aria-label="Delete conversation"
                    title="Delete"
                  >
                    <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                      <path d="M2 3.5h10M5.5 3.5V2h3v1.5M11.5 3.5l-.5 8.5h-8L2.5 3.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </button>
                </div>
              ))}
            </div>
          ))
        )}
      </nav>

      <div className="sidebar-footer">
        <button
          className="dark-mode-btn"
          onClick={onToggleDark}
          aria-label={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          {darkMode ? (
            <>
              <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
                <circle cx="7.5" cy="7.5" r="3.5" stroke="currentColor" strokeWidth="1.3"/>
                <path d="M7.5 1v1.5M7.5 12.5V14M1 7.5h1.5M12.5 7.5H14M3 3l1 1M11 11l1 1M3 12l1-1M11 4l1-1" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/>
              </svg>
              Light mode
            </>
          ) : (
            <>
              <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
                <path d="M13 7.5A6 6 0 116 1c-1 3 1.5 6 4 6.5.8.2 1.7.2 3 0z" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round"/>
              </svg>
              Dark mode
            </>
          )}
        </button>
      </div>
    </aside>
  );
}
