import { useState, useCallback, useRef, useEffect } from "react";
import { Chessboard } from "react-chessboard";
import { Chess } from "chess.js";
import { analyzePosition, analyzeEloRange, AnalyzeResponse, EloCurvePoint } from "./api";
import { MoveTable } from "./components/MoveTable";
import { EvalBar } from "./components/EvalBar";
import { EloCurveChart } from "./components/EloCurveChart";
import { DifficultyBadge } from "./components/DifficultyBadge";

const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

export default function App() {
  const [fen, setFen] = useState(START_FEN);
  const [fenInput, setFenInput] = useState(START_FEN);
  const [elo, setElo] = useState(1500);
  const [analysis, setAnalysis] = useState<AnalyzeResponse | null>(null);
  const [eloCurve, setEloCurve] = useState<EloCurvePoint[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const gameRef = useRef(new Chess());
  const debounceRef = useRef<ReturnType<typeof setTimeout>>();

  const analyze = useCallback(async (currentFen: string, currentElo: number) => {
    setLoading(true);
    setError(null);
    try {
      const [result, curve] = await Promise.all([
        analyzePosition(currentFen, currentElo),
        analyzeEloRange(currentFen),
      ]);
      setAnalysis(result);
      setEloCurve(curve);
    } catch (e: any) {
      setError(e.message || "Analysis failed");
    } finally {
      setLoading(false);
    }
  }, []);

  const handleEloChange = useCallback(
    (newElo: number) => {
      setElo(newElo);
      clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => analyze(fen, newElo), 300);
    },
    [fen, analyze]
  );

  const handleFenSubmit = useCallback(() => {
    try {
      const game = new Chess(fenInput);
      gameRef.current = game;
      setFen(fenInput);
      analyze(fenInput, elo);
    } catch {
      setError("Invalid FEN");
    }
  }, [fenInput, elo, analyze]);

  const onDrop = useCallback(
    (source: string, target: string) => {
      const game = gameRef.current;
      const move = game.move({ from: source, to: target, promotion: "q" });
      if (!move) return false;

      const newFen = game.fen();
      setFen(newFen);
      setFenInput(newFen);
      analyze(newFen, elo);
      return true;
    },
    [elo, analyze]
  );

  return (
    <div style={{ maxWidth: 1200, margin: "0 auto", padding: 20 }}>
      <h1 style={{ textAlign: "center", marginBottom: 20, color: "#e94560" }}>
        Chess Human Eval
      </h1>

      <div style={{ display: "flex", gap: 24, flexWrap: "wrap" }}>
        {/* Left: Board */}
        <div style={{ flex: "0 0 400px" }}>
          <Chessboard
            position={fen}
            onPieceDrop={onDrop}
            boardWidth={400}
            customDarkSquareStyle={{ backgroundColor: "#16213e" }}
            customLightSquareStyle={{ backgroundColor: "#0f3460" }}
          />

          {/* FEN Input */}
          <div style={{ marginTop: 12, display: "flex", gap: 8 }}>
            <input
              type="text"
              value={fenInput}
              onChange={(e) => setFenInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleFenSubmit()}
              style={{
                flex: 1, padding: 8, borderRadius: 4,
                border: "1px solid #333", background: "#16213e", color: "#e0e0e0",
                fontFamily: "monospace", fontSize: 12,
              }}
            />
            <button
              onClick={handleFenSubmit}
              style={{
                padding: "8px 16px", borderRadius: 4, border: "none",
                background: "#e94560", color: "white", cursor: "pointer",
              }}
            >
              Set
            </button>
          </div>

          {/* Elo Slider */}
          <div style={{ marginTop: 16 }}>
            <label style={{ display: "flex", justifyContent: "space-between" }}>
              <span>Player Elo</span>
              <strong>{elo}</strong>
            </label>
            <input
              type="range"
              min={800}
              max={2800}
              step={50}
              value={elo}
              onChange={(e) => handleEloChange(Number(e.target.value))}
              style={{ width: "100%", marginTop: 4 }}
            />
          </div>
        </div>

        {/* Right: Analysis */}
        <div style={{ flex: 1, minWidth: 300 }}>
          {error && (
            <div style={{ padding: 12, background: "#5c1a1a", borderRadius: 4, marginBottom: 12 }}>
              {error}
            </div>
          )}

          {loading && <p>Analyzing...</p>}

          {analysis && (
            <>
              <div style={{ display: "flex", gap: 16, marginBottom: 16, alignItems: "center" }}>
                <EvalBar
                  humanEval={analysis.human_eval}
                  engineEval={analysis.engine_eval}
                  valueHeadEval={analysis.value_head_eval}
                />
                {analysis.difficulty && (
                  <DifficultyBadge difficulty={analysis.difficulty} />
                )}
              </div>

              <MoveTable moves={analysis.top_moves} />
            </>
          )}

          {eloCurve && (
            <div style={{ marginTop: 24 }}>
              <h3 style={{ marginBottom: 8 }}>Eval vs Elo</h3>
              <EloCurveChart data={eloCurve} currentElo={elo} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
