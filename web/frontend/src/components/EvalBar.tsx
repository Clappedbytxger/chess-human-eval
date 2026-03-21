interface Props {
  humanEval: number | null;
  engineEval: number | null;
  valueHeadEval: number;
}

export function EvalBar({ humanEval, engineEval, valueHeadEval }: Props) {
  // Use value head eval as fallback display
  const displayEval = humanEval ?? valueHeadEval;
  // Map from [-1, 1] to percentage (0.5 = equal)
  const pct = Math.max(5, Math.min(95, (displayEval + 1) * 50));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <div style={{ display: "flex", gap: 12 }}>
        <EvalValue label="Human" value={displayEval} isRaw={humanEval === null} />
        {engineEval !== null && (
          <EvalValue label="Engine" value={engineEval / 100} isRaw={false} />
        )}
      </div>

      <div
        style={{
          width: 200,
          height: 20,
          background: "#333",
          borderRadius: 4,
          overflow: "hidden",
          position: "relative",
        }}
      >
        <div
          style={{
            position: "absolute",
            left: 0,
            top: 0,
            bottom: 0,
            width: `${pct}%`,
            background: "white",
            transition: "width 0.3s",
          }}
        />
        <div
          style={{
            position: "absolute",
            left: "50%",
            top: 0,
            bottom: 0,
            width: 2,
            background: "#666",
          }}
        />
      </div>
    </div>
  );
}

function EvalValue({
  label,
  value,
  isRaw,
}: {
  label: string;
  value: number;
  isRaw: boolean;
}) {
  const formatted = isRaw
    ? value.toFixed(2)
    : `${value > 0 ? "+" : ""}${value.toFixed(2)}`;

  return (
    <div>
      <div style={{ fontSize: 11, color: "#888", textTransform: "uppercase" }}>
        {label}
      </div>
      <div style={{ fontSize: 20, fontWeight: "bold", fontFamily: "monospace" }}>
        {formatted}
      </div>
    </div>
  );
}
