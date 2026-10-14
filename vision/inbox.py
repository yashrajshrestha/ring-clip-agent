"""Inbox: drop a clip into a folder and it gets analysed and decided.

    python -m vision.inbox [--inbox clips/inbox] [--out web/public/samples] [--interval 2]

Polls the folder. A new video whose size has stopped changing between two
polls is transcoded, analysed (OpenCV, detector), decided (Decider 2B when
DECIDER_URL is set, else the rule stub; recheck and escalation as configured)
and written to the samples folder, where the playground picks it up. A file
touched or replaced after its timeline was written is processed again, which
is how the playground's Re-run button works. A sidecar `<name>.profile` next
to the video names an analysis profile for that run ("harder" for the Look
harder button) and is removed once used. Status is kept in <out>/inbox.json.
The playground runs this watcher in a thread, so `python -m vision.playground`
alone gives a working drop box.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
from pathlib import Path

from .profiles import PROFILES
from .timeline import VIDEO_EXT, load_detector, process_clip


class Inbox:
    def __init__(self, inbox: Path, out_dir: Path, detector=None, interval: float = 2.0) -> None:
        self.inbox, self.out_dir, self.detector, self.interval = Path(inbox), Path(out_dir), detector, interval
        self.inbox.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # finished: stem -> time it last completed, so a page can tell a re-run apart from an old run
        self.status: dict = {"processing": None, "queue": [], "done": [], "finished": {}, "errors": {}}
        self._sizes: dict[str, int] = {}   # last seen size per file, to wait out a copy in progress
        self._write_status()

    def _write_status(self) -> None:
        (self.out_dir / "inbox.json").write_text(json.dumps({**self.status, "now": time.time()}))

    def pending(self) -> list[Path]:
        """Videos with no timeline yet, or changed since their timeline was written."""
        out = []
        for p in sorted(self.inbox.iterdir()):
            if p.suffix.lower() not in VIDEO_EXT or p.name.startswith(".") or not p.is_file():
                continue
            tl = self.out_dir / f"{p.stem}.timeline.json"
            if not tl.exists() or tl.stat().st_mtime < p.stat().st_mtime:
                out.append(p)
        return out

    def _stable(self, p: Path) -> bool:
        """True when the file's size has not changed since the previous poll."""
        size = p.stat().st_size
        stable = self._sizes.get(p.name) == size and size > 0
        self._sizes[p.name] = size
        return stable

    def _profile_for(self, p: Path) -> str:
        """The profile named by a sidecar file, consumed on read; `normal` otherwise."""
        sidecar = p.with_suffix(".profile")
        if not sidecar.exists():
            return "normal"
        name = sidecar.read_text().strip()
        sidecar.unlink()
        return name if name in PROFILES else "normal"

    def poll(self) -> list[str]:
        """One pass: process every stable pending clip. Returns the names processed."""
        pending = self.pending()
        self.status["queue"] = [p.stem for p in pending]
        self._write_status()
        done = []
        for p in pending:
            if not self._stable(p):
                continue
            self.status["processing"] = p.stem
            self.status["queue"] = [q.stem for q in pending if q != p]
            self.status["progress"] = {"stage": "convert", "done": 0, "total": 1}
            self._write_status()
            last = [0.0]

            def progress(stage: str, done: int, total: int) -> None:
                self.status["progress"] = {"stage": stage, "done": done, "total": total}
                if stage != "analyse" or time.time() - last[0] > 0.5:   # frames tick fast; don't rewrite the file each time
                    last[0] = time.time()
                    self._write_status()

            try:
                process_clip(p, self.out_dir, self.detector, self._profile_for(p), progress)
                if p.stem not in self.status["done"]:
                    self.status["done"].append(p.stem)
                self.status["finished"][p.stem] = time.time()
                self.status["errors"].pop(p.stem, None)
                done.append(p.stem)
            except Exception as e:  # noqa: BLE001  one bad file must not stop the watcher
                self.status["errors"][p.stem] = f"{type(e).__name__}: {e}"
                traceback.print_exc()
                # mark it handled so the same broken file is not retried every poll
                (self.out_dir / f"{p.stem}.timeline.json").touch()
            self.status["processing"] = None
            self.status["progress"] = None
            self._write_status()
        return done

    def run(self, stop: threading.Event | None = None) -> None:
        while stop is None or not stop.is_set():
            self.poll()
            time.sleep(self.interval)

    def start_thread(self) -> threading.Event:
        """Run the watcher in a daemon thread; set the returned event to stop it."""
        stop = threading.Event()
        threading.Thread(target=self.run, args=(stop,), daemon=True, name="inbox").start()
        return stop


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inbox", type=Path, default=Path("clips/inbox"))
    ap.add_argument("--out", type=Path, default=Path("web/public/samples"))
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--no-detect", action="store_true")
    args = ap.parse_args()
    box = Inbox(args.inbox, args.out, None if args.no_detect else load_detector(), args.interval)
    print(f"watching {args.inbox} -> {args.out}", file=sys.stderr)
    box.run()


if __name__ == "__main__":
    main()
