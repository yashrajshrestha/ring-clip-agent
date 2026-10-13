import { useCallback, useEffect, useRef, useState } from "react";
import type { Box, ClipEvent, Decision, Frame, Timeline } from "./types";

const SCENE_SHOW_SECONDS = 4; // keep a "left / taken" box on screen this long

const ACTION_PHRASE: Record<Decision["action"], string> = {
  alert: "Alert sent.",
  ignore: "Ignored.",
  recheck: "Needs a recheck.",
  log: "Logged.",
};

// Largest index whose timestamp is at or before t. Lookup is by time, not
// frame number, because Ring clips drop frames.
function frameAt(frames: Frame[], t: number): number {
  let lo = 0;
  let hi = frames.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (frames[mid].t <= t + 1e-3) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

function fmt(t: number): string {
  const m = Math.floor(t / 60);
  const s = (t % 60).toFixed(1).padStart(4, "0");
  return `${m}:${s}`;
}

function cssVar(el: Element, name: string): string {
  return getComputedStyle(el).getPropertyValue(name).trim();
}

export default function Player({ tl, base }: { tl: Timeline; base: string }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const spoken = useRef<Set<number>>(new Set());
  const lastT = useRef(0);
  const [voice, setVoice] = useState(true);
  const voiceRef = useRef(voice);
  const [now, setNow] = useState(0);
  const [frameIdx, setFrameIdx] = useState(0);

  useEffect(() => {
    voiceRef.current = voice;
    if (!voice) window.speechSynthesis?.cancel();
  }, [voice]);

  const narrate = useCallback(
    (t: number, playing: boolean) => {
      // On a backward seek, forget what was said after the new position.
      if (t < lastT.current - 0.25) {
        spoken.current = new Set(tl.events.filter((e) => e.t_start < t).map((e) => e.id));
      }
      lastT.current = t;
      for (const ev of tl.events) {
        if (spoken.current.has(ev.id) || ev.t_start > t) continue;
        spoken.current.add(ev.id);
        // Speak only events crossed during playback, not ones skipped by a seek.
        if (playing && voiceRef.current && t - ev.t_start < 1.0 && "speechSynthesis" in window) {
          const u = new SpeechSynthesisUtterance(`${ev.say} ${ACTION_PHRASE[ev.decision.action]}`);
          u.rate = 1.1;
          window.speechSynthesis.speak(u);
        }
      }
    },
    [tl],
  );

  const draw = useCallback(
    (t: number) => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas) return;
      const dpr = window.devicePixelRatio || 1;
      const w = Math.round(canvas.clientWidth * dpr);
      const h = Math.round(canvas.clientHeight * dpr);
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      const ctx = canvas.getContext("2d")!;
      ctx.clearRect(0, 0, w, h);

      const box = (b: Box, colour: string, label: string) => {
        const [x, y, bw, bh] = [b[0] * w, b[1] * h, b[2] * w, b[3] * h];
        ctx.lineWidth = 3 * dpr;
        ctx.strokeStyle = colour;
        ctx.strokeRect(x, y, bw, bh);
        ctx.font = `${13 * dpr}px system-ui, sans-serif`;
        const tw = ctx.measureText(label).width + 10 * dpr;
        const ly = Math.max(0, y - 20 * dpr);
        ctx.fillStyle = colour;
        ctx.fillRect(x, ly, tw, 20 * dpr);
        ctx.fillStyle = "#fff";
        ctx.fillText(label, x + 5 * dpr, ly + 14 * dpr);
      };

      const i = frameAt(tl.frames, t);
      const f = tl.frames[i];
      // YOLOX labels when present; the raw motion box only when nothing was recognised.
      const objs = (f?.objs ?? []).filter((o) => o.group !== "other");
      for (const o of objs) {
        box(o.box, cssVar(canvas, o.group === "person" ? "--alert" : "--info"), `${o.label} ${Math.round(o.score * 100)}%`);
      }
      if (f?.box && objs.length === 0) box(f.box, cssVar(canvas, "--alert"), "motion");
      for (const ev of tl.events) {
        if (ev.type === "scene_change" && ev.box && t >= ev.t_start && t < ev.t_start + SCENE_SHOW_SECONDS) {
          box(ev.box, cssVar(canvas, "--warn"), "left / taken");
        }
      }

      setNow(t);
      setFrameIdx(i);
      narrate(t, !video.paused);
    },
    [tl, narrate],
  );

  useEffect(() => {
    const v = videoRef.current!;
    let stopped = false;
    let handle = 0;
    const hasRVFC = "requestVideoFrameCallback" in HTMLVideoElement.prototype;
    if (hasRVFC) {
      // mediaTime is the presentation time of the frame on screen, so the
      // overlay matches the exact frame rather than the clock.
      const cb = (_: number, md: VideoFrameCallbackMetadata) => {
        if (stopped) return;
        draw(md.mediaTime);
        handle = v.requestVideoFrameCallback(cb);
      };
      handle = v.requestVideoFrameCallback(cb);
    } else {
      const loop = () => {
        if (stopped) return;
        draw(v.currentTime);
        handle = requestAnimationFrame(loop);
      };
      handle = requestAnimationFrame(loop);
    }
    const redraw = () => draw(v.currentTime);
    v.addEventListener("seeked", redraw);
    v.addEventListener("loadeddata", redraw);
    window.addEventListener("resize", redraw);
    return () => {
      stopped = true;
      if (hasRVFC) v.cancelVideoFrameCallback(handle);
      else cancelAnimationFrame(handle);
      v.removeEventListener("seeked", redraw);
      v.removeEventListener("loadeddata", redraw);
      window.removeEventListener("resize", redraw);
      window.speechSynthesis?.cancel();
    };
  }, [draw]);

  const seek = (t: number) => {
    const v = videoRef.current;
    if (v) v.currentTime = Math.max(0, Math.min(t, tl.duration));
  };

  const f = tl.frames[frameIdx];
  const fired = tl.events.filter((e) => e.t_start <= now + 1e-3).reverse();
  const blurPct = Math.min(100, (f.blur / (tl.thresholds.blur_min * 10)) * 100);
  const blurLow = f.blur < tl.thresholds.blur_min;

  return (
    <div className="player">
      <section className="stage">
        <div className="frame" style={{ aspectRatio: `${tl.width} / ${tl.height}` }}>
          <video ref={videoRef} src={base + tl.video} controls playsInline preload="auto" />
          <canvas ref={canvasRef} />
        </div>

        <div className="status">
          <span className={`badge ${f.ir ? "on" : ""}`}>{f.ir ? "Night IR" : "Day colour"}</span>
          <span className="meter" title={`Laplacian variance ${f.blur}`}>
            Sharpness
            <span className="bar">
              <span className={blurLow ? "fill low" : "fill"} style={{ width: `${blurPct}%` }} />
            </span>
            <b>{blurLow ? "blurry" : "ok"}</b>
          </span>
          <span className="meter">
            Brightness <b>{Math.round(f.bright)}</b>
          </span>
          <span className="clock">{fmt(now)}</span>
          <label className="toggle">
            <input type="checkbox" checked={voice} onChange={(e) => setVoice(e.target.checked)} /> Voice
          </label>
        </div>

        <div
          className="track"
          onClick={(e) => {
            const r = e.currentTarget.getBoundingClientRect();
            seek(((e.clientX - r.left) / r.width) * tl.duration);
          }}
        >
          {tl.events.map((ev) => (
            <span
              key={ev.id}
              className={`mark ${ev.decision.action}`}
              title={`${fmt(ev.t_start)} ${ev.say}`}
              style={{
                left: `${(ev.t_start / tl.duration) * 100}%`,
                width: `max(6px, ${((ev.t_end - ev.t_start) / tl.duration) * 100}%)`,
              }}
            />
          ))}
          <span className="head" style={{ left: `${(now / tl.duration) * 100}%` }} />
        </div>
      </section>

      <aside className="log">
        <h2>Live log</h2>
        {fired.length === 0 && <p className="empty">Press play. Events appear as the clip reaches them.</p>}
        <ol>
          {fired.map((ev: ClipEvent) => (
            <li key={ev.id} onClick={() => seek(ev.t_start)}>
              <div className="row">
                <span className="time">{fmt(ev.t_start)}</span>
                <span className={`pill ${ev.decision.action}`}>{ev.decision.action}</span>
              </div>
              <div className="say">{ev.say}</div>
              {ev.objects && Object.keys(ev.objects).length > 0 && (
                <div className="reason">
                  YOLOX:{" "}
                  {Object.entries(ev.objects)
                    .map(([g, n]) => `${g} ×${n}`)
                    .join(", ")}
                </div>
              )}
              <div className="reason">
                {ev.decision.reason}
                <span className="src">
                  {ev.decision.source}
                  {ev.decision.confidence != null && ` · ${Math.round(ev.decision.confidence * 100)}% sure`}
                </span>
              </div>
            </li>
          ))}
        </ol>
      </aside>
    </div>
  );
}
