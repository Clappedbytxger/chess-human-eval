interface Props {
  difficulty: string;
}

const colors: Record<string, { bg: string; text: string }> = {
  easy: { bg: "#1a3a2a", text: "#4ecca3" },
  tricky: { bg: "#3a3a1a", text: "#e9c46a" },
  deadly: { bg: "#3a1a1a", text: "#e94560" },
};

export function DifficultyBadge({ difficulty }: Props) {
  const style = colors[difficulty] ?? colors.tricky;

  return (
    <span
      style={{
        padding: "4px 12px",
        borderRadius: 12,
        background: style.bg,
        color: style.text,
        fontWeight: "bold",
        fontSize: 14,
        textTransform: "uppercase",
        letterSpacing: 1,
      }}
    >
      {difficulty}
    </span>
  );
}
