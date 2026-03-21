import type { MoveInfo } from "../api";

interface Props {
  moves: MoveInfo[];
}

export function MoveTable({ moves }: Props) {
  if (!moves.length) return <p>No moves to display</p>;

  return (
    <div>
      <h3 style={{ marginBottom: 8 }}>Top Moves</h3>
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr style={{ borderBottom: "2px solid #333" }}>
            <th style={thStyle}>#</th>
            <th style={thStyle}>Move</th>
            <th style={{ ...thStyle, textAlign: "right" }}>Probability</th>
            {moves.some((m) => m.eval !== null) && (
              <th style={{ ...thStyle, textAlign: "right" }}>Eval</th>
            )}
          </tr>
        </thead>
        <tbody>
          {moves.map((move, i) => (
            <tr key={move.move_uci} style={{ borderBottom: "1px solid #222" }}>
              <td style={tdStyle}>{i + 1}</td>
              <td style={{ ...tdStyle, fontWeight: "bold" }}>{move.move}</td>
              <td style={{ ...tdStyle, textAlign: "right" }}>
                <span style={{ color: probColor(move.probability) }}>
                  {(move.probability * 100).toFixed(1)}%
                </span>
                <div style={barContainerStyle}>
                  <div
                    style={{
                      ...barStyle,
                      width: `${move.probability * 100}%`,
                      background: probColor(move.probability),
                    }}
                  />
                </div>
              </td>
              {move.eval !== null && (
                <td style={{ ...tdStyle, textAlign: "right", fontFamily: "monospace" }}>
                  {move.eval! > 0 ? "+" : ""}
                  {(move.eval! / 100).toFixed(2)}
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function probColor(p: number): string {
  if (p > 0.3) return "#4ecca3";
  if (p > 0.1) return "#e9c46a";
  return "#e76f51";
}

const thStyle: React.CSSProperties = {
  padding: "8px 12px",
  textAlign: "left",
  color: "#999",
  fontSize: 12,
  textTransform: "uppercase",
};

const tdStyle: React.CSSProperties = {
  padding: "8px 12px",
};

const barContainerStyle: React.CSSProperties = {
  height: 4,
  background: "#222",
  borderRadius: 2,
  marginTop: 4,
};

const barStyle: React.CSSProperties = {
  height: "100%",
  borderRadius: 2,
  transition: "width 0.3s",
};
