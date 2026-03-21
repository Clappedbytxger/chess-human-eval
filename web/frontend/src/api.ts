const API_BASE = "/api";

export interface MoveInfo {
  move: string;
  move_uci: string;
  probability: number;
  eval: number | null;
  contribution: number | null;
}

export interface AnalyzeResponse {
  fen: string;
  elo: number;
  top_moves: MoveInfo[];
  human_eval: number | null;
  engine_eval: number | null;
  eval_gap: number | null;
  difficulty: string | null;
  value_head_eval: number;
}

export interface EloCurvePoint {
  elo: number;
  human_eval: number;
  difficulty: string;
}

export async function analyzePosition(
  fen: string,
  elo: number,
  topK: number = 5
): Promise<AnalyzeResponse> {
  const res = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ fen, elo, top_k: topK }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function analyzeEloRange(
  fen: string,
  eloMin: number = 800,
  eloMax: number = 2800,
  eloStep: number = 100
): Promise<EloCurvePoint[]> {
  const res = await fetch(`${API_BASE}/analyze-elo-range`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ fen, elo_min: eloMin, elo_max: eloMax, elo_step: eloStep }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
