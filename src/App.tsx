import { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { ChatPanel } from './components/ChatPanel';
import { useChat } from './hooks/useChat';

export function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [darkMode, setDarkMode] = useState(() => {
    return localStorage.getItem('hcsa_theme') === 'dark';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', darkMode ? 'dark' : 'light');
    localStorage.setItem('hcsa_theme', darkMode ? 'dark' : 'light');
  }, [darkMode]);

  const {
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
  } = useChat();

  return (
    <div className={`app-root ${sidebarOpen ? 'sidebar-visible' : 'sidebar-hidden'}`}>
      {/* Sidebar overlay on mobile */}
      {sidebarOpen && (
        <div
          className="sidebar-overlay"
          onClick={() => setSidebarOpen(false)}
          aria-hidden="true"
        />
      )}

      <Sidebar
        conversations={conversations}
        activeId={activeId}
        onSelect={selectConversation}
        onNew={startNewChat}
        onDelete={removeConversation}
        onToggle={() => setSidebarOpen((v) => !v)}
        darkMode={darkMode}
        onToggleDark={() => setDarkMode((v) => !v)}
      />

      <main className="main-area">
        {/* Top bar (mobile) */}
        <div className="topbar">
          <button
            className="topbar-menu-btn"
            onClick={() => setSidebarOpen((v) => !v)}
            aria-label="Toggle sidebar"
          >
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <path d="M3 5h14M3 10h14M3 15h14" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round"/>
            </svg>
          </button>
          <span className="topbar-title">
            {activeConversation ? activeConversation.title : 'HCSA Assistant'}
          </span>
          <button
            className="topbar-new-btn"
            onClick={startNewChat}
            aria-label="New chat"
          >
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
              <path d="M9 2v14M2 9h14" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </button>
        </div>

        <ChatPanel
          messages={activeConversation?.messages ?? []}
          isLoading={isLoading}
          streamedText={streamedText}
          statusLine={statusLine}
          onSubmit={submitMessage}
        />
      </main>
    </div>
  );
}
