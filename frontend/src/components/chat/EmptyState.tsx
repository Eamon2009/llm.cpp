export function EmptyState() {
  return (
    <div
      style={{
        flex: 1,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "24px",
      }}
    >
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 20,
          textAlign: "center",
          maxWidth: 420,
        }}
      >
        {/* Icon */}
        <div
          style={{
            width: 56,
            height: 56,
            borderRadius: 14,
            background: "linear-gradient(135deg, #4f8ef7 0%, #2563eb 100%)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            boxShadow: "0 8px 32px rgba(79,142,247,0.25)",
          }}
        >
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
            <path
              d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 14H9V8h2v8zm4 0h-2V8h2v8z"
              fill="white"
              opacity="0.9"
            />
            <path
              d="M8 12l2-2 2 2 4-4"
              stroke="white"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              fill="none"
            />
          </svg>
        </div>

        <div>
          <h1
            style={{
              margin: 0,
              fontSize: 20,
              fontWeight: 600,
              color: "var(--text-primary)",
              letterSpacing: "-0.3px",
            }}
          >
            Quadtrix.cpp
          </h1>
          <p
            style={{
              margin: "8px 0 0",
              fontSize: 13,
              color: "var(--text-muted)",
              lineHeight: 1.6,
            }}
          >
            Local char-level language model. Start a conversation below.
          </p>
        </div>

        {/* Hint chips */}
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8, justifyContent: "center" }}>
          {["Fast local inference", "C++ & PyTorch backends", "No cloud required"].map((chip) => (
            <span
              key={chip}
              style={{
                padding: "4px 10px",
                borderRadius: 20,
                border: "1px solid var(--border-muted)",
                fontSize: 11,
                color: "var(--text-muted)",
                background: "var(--bg-elevated)",
              }}
            >
              {chip}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
