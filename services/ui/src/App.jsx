import { useState, useEffect, useCallback } from "react";
import axios from "axios";
import Stats from "./components/Stats";
import RecommendPanel from "./components/RecommendPanel";
import FeedbackPanel from "./components/FeedbackPanel";

const INFERENCE = "http://localhost:8002";
const FEEDBACK  = "http://localhost:8001";
const THRESHOLD = 50;

export default function App() {
  const [userId, setUserId]       = useState(1);
  const [movies, setMovies]       = useState([]);
  const [loading, setLoading]     = useState(false);
  const [totalEvents, setTotal]   = useState(0);
  const [baseline, setBaseline]   = useState(0);
  const [modelReady, setReady]    = useState(false);
  const [sessionCount, setSession]= useState(0);
  const [log, setLog]             = useState([]);
  const [actions, setActions]     = useState({});

  const addLog = (msg, type = "") => {
    const time = new Date().toLocaleTimeString();
    setLog(prev => [{ msg, type, time }, ...prev].slice(0, 40));
  };

  const fetchStats = useCallback(async () => {
    try {
      const [s, m] = await Promise.all([
        axios.get(`${FEEDBACK}/feedback/stats`),
        axios.get(`${INFERENCE}/model/status`),
      ]);
      setTotal(s.data.total_events || 0);
      setReady(m.data.model_loaded);
    } catch {
      setReady(false);
    }
  }, []);

  useEffect(() => {
    fetchStats();
    const id = setInterval(fetchStats, 5000);
    return () => clearInterval(id);
  }, [fetchStats]);

  const getRecommendations = async () => {
    setLoading(true);
    setMovies([]);
    setActions({});
    try {
      const res = await axios.get(`${INFERENCE}/recommend/${userId}`);
      setMovies(res.data.recommendations || []);
      addLog(`Got ${res.data.recommendations.length} recs for user ${userId}`, "success");
    } catch {
      addLog("Could not reach inference API", "error");
    }
    setLoading(false);
  };

  const sendFeedback = async (type, movieId, extra = {}) => {
    const bodies = {
      rating: { user_id: userId, movie_id: movieId, rating: extra.rating },
      click:  { user_id: userId, movie_id: movieId, position: extra.position },
      skip:   { user_id: userId, movie_id: movieId },
    };
    try {
      await axios.post(`${FEEDBACK}/feedback/${type}`, bodies[type]);
      setActions(prev => ({ ...prev, [movieId]: type }));
      setSession(s => s + 1);
      const label = type === "rating" ? `rated ${extra.rating}★` : type;
      addLog(`User ${userId} ${label} → movie ${movieId}`, type === "skip" ? "warn" : "success");
      fetchStats();
    } catch {
      addLog(`Failed to send ${type}`, "error");
    }
  };

  const newEvents = Math.max(0, totalEvents - baseline);
  const pct       = Math.min(100, Math.round((newEvents / THRESHOLD) * 100));
  const remaining = Math.max(0, THRESHOLD - newEvents);

  return (
    <div>
      <header style={{ background:"#1a1a1a", color:"white", padding:"14px 32px", display:"flex", justifyContent:"space-between", alignItems:"center" }}>
        <div>
          <span style={{ fontWeight:500, fontSize:16 }}>Data Flywheel</span>
          <span style={{ color:"#666", fontSize:13, marginLeft:12 }}>Movie Recommender</span>
        </div>
        <div style={{ display:"flex", alignItems:"center", gap:8 }}>
          <span style={{ width:8, height:8, borderRadius:"50%", background: modelReady ? "#4ade80" : "#f87171", display:"inline-block" }}/>
          <span style={{ fontSize:12, color:"#aaa" }}>{modelReady ? "Model ready" : "Fallback mode"}</span>
        </div>
      </header>

      <div style={{ maxWidth:900, margin:"0 auto", padding:"28px 20px" }}>
        <Stats total={totalEvents} remaining={remaining} modelReady={modelReady} session={sessionCount} />
        <RecommendPanel
          userId={userId} setUserId={setUserId}
          movies={movies} loading={loading} actions={actions}
          onRecommend={getRecommendations}
          onFeedback={sendFeedback}
          onRandom={() => { setUserId(Math.floor(Math.random()*610)+1); }}
        />
        <FeedbackPanel log={log} pct={pct} newEvents={newEvents} threshold={THRESHOLD} />
      </div>
    </div>
  );
}
