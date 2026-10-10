import { useEffect, useState } from "react";
import Player from "./Player";
import type { IndexEntry, Timeline } from "./types";

const BASE = "/samples/";

export default function App() {
  const [index, setIndex] = useState<IndexEntry[]>([]);
  const [selected, setSelected] = useState<string>("");
  const [tl, setTl] = useState<Timeline | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    fetch(BASE + "index.json")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`index.json ${r.status}`))))
      .then((rows: IndexEntry[]) => {
        setIndex(rows);
        if (rows.length) setSelected(rows[0].timeline);
      })
      .catch((e) => {
        console.error(e);
        setFailed(true);
      });
  }, []);

  useEffect(() => {
    if (!selected) return;
    setTl(null);
    fetch(BASE + selected)
      .then((r) => r.json())
      .then(setTl)
      .catch((e) => {
        console.error(e);
        setFailed(true);
      });
  }, [selected]);

  return (
    <main>
      <header>
        <div>
          <h1>Ring Clip Agent</h1>
          <p className="sub">OpenCV 5 checks run on the clip, the decider explains each event, the page narrates in sync.</p>
        </div>
        {index.length > 0 && (
          <select value={selected} onChange={(e) => setSelected(e.target.value)} aria-label="Clip">
            {index.map((row) => (
              <option key={row.timeline} value={row.timeline}>
                {row.name} ({row.events} events, {row.duration.toFixed(0)}s)
              </option>
            ))}
          </select>
        )}
      </header>
      {failed && (
        <p className="error">
          No clip data found. Run the worker first: <code>docker compose run --rm worker</code>
        </p>
      )}
      {tl && <Player key={tl.clip} tl={tl} base={BASE} />}
    </main>
  );
}
