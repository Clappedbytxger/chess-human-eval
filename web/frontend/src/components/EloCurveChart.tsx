import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import type { EloCurvePoint } from "../api";

interface Props {
  data: EloCurvePoint[];
  currentElo: number;
}

export function EloCurveChart({ data, currentElo }: Props) {
  return (
    <ResponsiveContainer width="100%" height={250}>
      <LineChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#333" />
        <XAxis
          dataKey="elo"
          stroke="#888"
          fontSize={12}
          tickFormatter={(v) => `${v}`}
        />
        <YAxis stroke="#888" fontSize={12} />
        <Tooltip
          contentStyle={{ background: "#1a1a2e", border: "1px solid #333" }}
          labelFormatter={(v) => `Elo ${v}`}
        />
        <ReferenceLine x={currentElo} stroke="#e94560" strokeDasharray="5 5" />
        <Line
          type="monotone"
          dataKey="human_eval"
          stroke="#4ecca3"
          strokeWidth={2}
          dot={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
