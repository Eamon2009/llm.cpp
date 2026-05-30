export function ThinkingIndicator() {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, color: "var(--text-muted)" }}>
      <span style={{ fontSize: 12 }}>Generating</span>
      <span style={{ display: "flex", gap: 3 }}>
        {[0, 120, 240].map((delay) => (
          <span
            key={delay}
            style={{
              display: "inline-block",
              width: 5,
              height: 5,
              borderRadius: "50%",
              background: "var(--accent)",
              animation: `bounce 1s ease-in-out ${delay}ms infinite`,
            }}
          />
        ))}
      </span>
      <style>{`
        @keyframes bounce {
          0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
          40% { transform: translateY(-4px); opacity: 1; }
        }
      `}</style>
    </div>
  );
}
