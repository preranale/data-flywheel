export default function FeedbackPanel({ log, pct, newEvents, threshold }) {
  const colors = { success:"#16a34a", error:"#dc2626", warn:"#d97706", "":"#555" };
  return (
    <div style={{ background:"white", borderRadius:12, border:"0.5px solid #e5e5e0" }}>
      <div style={{ padding:"14px 20px", borderBottom:"0.5px solid #e5e5e0", display:"flex", justifyContent:"space-between", alignItems:"center" }}>
        <span style={{ fontWeight:500, fontSize:14 }}>Feedback stream</span>
        <span style={{ fontSize:12, color:"#aaa" }}>threshold: {threshold} new events</span>
      </div>
      <div style={{ padding:20 }}>
        <div style={{ display:"flex", justifyContent:"space-between", fontSize:12, color:"#888", marginBottom:6 }}>
          <span>{newEvents} / {threshold} events toward next retrain</span>
          <span>{pct}%</span>
        </div>
        <div style={{ height:6, background:"#eee", borderRadius:3, overflow:"hidden", marginBottom:16 }}>
          <div style={{ height:"100%", background: pct >= 100 ? "#d97706" : "#1a1a1a", width:`${pct}%`, borderRadius:3, transition:"width 0.5s" }}/>
        </div>
        <div style={{ fontFamily:"monospace", fontSize:12, background:"#f9f9f7", borderRadius:8, padding:12, maxHeight:180, overflowY:"auto" }}>
          {log.length === 0 && <div style={{ color:"#aaa" }}>Waiting for events...</div>}
          {log.map((e, i) => (
            <div key={i} style={{ color: colors[e.type] || "#555", borderBottom:"0.5px solid #f0f0ee", padding:"3px 0", lineHeight:1.8 }}>
              [{e.time}] {e.msg}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
