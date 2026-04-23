export default function RecommendPanel({ userId, setUserId, movies, loading, actions, onRecommend, onFeedback, onRandom }) {
  return (
    <div style={{ background:"white", borderRadius:12, border:"0.5px solid #e5e5e0", marginBottom:20 }}>
      <div style={{ padding:"14px 20px", borderBottom:"0.5px solid #e5e5e0", display:"flex", justifyContent:"space-between", alignItems:"center" }}>
        <span style={{ fontWeight:500, fontSize:14 }}>Get recommendations</span>
        <span style={{ fontSize:12, color:"#aaa" }}>SVD matrix factorisation</span>
      </div>
      <div style={{ padding:20 }}>
        <div style={{ display:"flex", gap:10, alignItems:"center", marginBottom:20 }}>
          <label style={{ fontSize:13, color:"#666" }}>User ID</label>
          <input
            type="number" min="1" max="610" value={userId}
            onChange={e => setUserId(parseInt(e.target.value))}
            style={{ border:"0.5px solid #ddd", borderRadius:8, padding:"8px 12px", fontSize:14, width:90, outline:"none" }}
          />
          <button onClick={onRecommend} style={btnStyle("#1a1a1a", "white")}>
            {loading ? "Loading..." : "Recommend"}
          </button>
          <button onClick={() => { onRandom(); setTimeout(onRecommend, 100); }} style={btnStyle("white", "#1a1a1a", "#ddd")}>
            Random user
          </button>
        </div>

        {loading && <div style={{ textAlign:"center", padding:32, color:"#aaa", fontSize:14 }}>Fetching recommendations...</div>}

        {!loading && movies.length === 0 && (
          <div style={{ textAlign:"center", padding:32, color:"#aaa", fontSize:14 }}>Enter a user ID and click Recommend</div>
        )}

        {!loading && movies.map((m, i) => (
          <div key={m.movie_id} style={{ display:"flex", alignItems:"center", justifyContent:"space-between", padding:"12px 16px", background:"#f9f9f7", borderRadius:8, border:"0.5px solid #eee", marginBottom:8 }}>
            <div style={{ display:"flex", alignItems:"center", gap:12 }}>
              <span style={{ fontSize:12, color:"#aaa", width:20 }}>{i+1}</span>
              <div>
                <div style={{ fontSize:14, fontWeight:500 }}>{m.title}</div>
                <div style={{ fontSize:12, color:"#888", marginTop:2 }}>Predicted: {m.predicted_rating}★</div>
              </div>
            </div>
            <div style={{ display:"flex", gap:6 }}>
              {actions[m.movie_id] ? (
                <span style={{ fontSize:12, color:"#888", padding:"5px 12px" }}>
                  {actions[m.movie_id] === "rating" ? "Rated" : actions[m.movie_id] === "click" ? "Clicked" : "Skipped"}
                </span>
              ) : (
                <>
                  <button onClick={() => onFeedback("click",  m.movie_id, { position: i+1 })} style={actionBtn}>Clicked</button>
                  <button onClick={() => onFeedback("rating", m.movie_id, { rating: 5 })}      style={{...actionBtn, color:"#16a34a", borderColor:"#16a34a"}}>Loved it</button>
                  <button onClick={() => onFeedback("rating", m.movie_id, { rating: 2 })}      style={actionBtn}>Meh</button>
                  <button onClick={() => onFeedback("skip",   m.movie_id)}                     style={{...actionBtn, color:"#dc2626", borderColor:"#dc2626"}}>Skip</button>
                </>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

const btnStyle = (bg, color, border) => ({
  padding:"8px 16px", borderRadius:8, border:`0.5px solid ${border || bg}`,
  background:bg, color, fontSize:13, cursor:"pointer"
});

const actionBtn = {
  padding:"5px 12px", borderRadius:6, border:"0.5px solid #ddd",
  background:"white", fontSize:12, cursor:"pointer", color:"#444"
};
