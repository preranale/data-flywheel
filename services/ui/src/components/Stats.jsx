export default function Stats({ total, remaining, modelReady, session }) {
  const cards = [
    { label: "Total events",    value: total,    color: "#2563eb" },
    { label: "Until retrain",   value: remaining === 0 ? "Now!" : remaining, color: remaining === 0 ? "#d97706" : "#1a1a1a" },
    { label: "Model",           value: modelReady ? "Ready" : "Fallback", color: modelReady ? "#16a34a" : "#dc2626" },
    { label: "Session actions", value: session,  color: "#1a1a1a" },
  ];
  return (
    <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:12, marginBottom:24 }}>
      {cards.map(c => (
        <div key={c.label} style={{ background:"white", borderRadius:12, padding:"14px 16px", border:"0.5px solid #e5e5e0" }}>
          <div style={{ fontSize:11, color:"#888", textTransform:"uppercase", letterSpacing:"0.5px", marginBottom:6 }}>{c.label}</div>
          <div style={{ fontSize:22, fontWeight:500, color:c.color }}>{c.value}</div>
        </div>
      ))}
    </div>
  );
}
