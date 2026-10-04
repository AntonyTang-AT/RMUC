"""Small Chromium probe for interactive replay startup and frame changes."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from playwright.sync_api import sync_playwright


def summarize(values):
    values = sorted(values)
    if not values:
        return {}
    return {
        "count": len(values),
        "median_ms": round(statistics.median(values), 2),
        "p95_ms": round(values[min(len(values) - 1, int(len(values) * .95))], 2),
        "max_ms": round(values[-1], 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("html", type=Path)
    parser.add_argument("--samples", type=int, default=45)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1200, "height": 900})
        start = time.perf_counter()
        page.goto(args.html.resolve().as_uri(), wait_until="load")
        page.wait_for_function("document.querySelector('#rm-frame')?.max !== '0'", timeout=60000)
        startup_ms = (time.perf_counter() - start) * 1000
        profile = page.evaluate("""(samples) => {
          const slider=document.querySelector('#rm-frame');
          const map=document.querySelector('#rm-map');
          const global=document.querySelector('#rm-global');
          const table=document.querySelector('#rm-eval');
          const count=Math.min(samples,Number(slider.max));
          const changes=[];
          for(let i=1;i<=count;i++){
            const begin=performance.now();
            slider.value=String(i);
            slider.dispatchEvent(new Event('input',{bubbles:true}));
            // Include layout work as well as the synchronous DOM update.
            map.getBoundingClientRect(); global.getBoundingClientRect(); table.getBoundingClientRect();
            changes.push(performance.now()-begin);
          }
          return {changes, frames:Number(slider.max)+1,
                  zone_nodes:document.querySelector('#rm-zone-layer').childElementCount,
                  marker_nodes:document.querySelector('#rm-markers').childElementCount};
        }""", args.samples)
        playback = page.evaluate("""async () => {
          const slider=document.querySelector('#rm-frame');
          const button=document.querySelector('#rm-play');
          const select=document.querySelector('#rm-interval');
          const defaultInterval=Number(select.value);
          async function run(interval){
            select.value=String(interval);
            slider.value='0';slider.dispatchEvent(new Event('input',{bubbles:true}));
            const start=performance.now();button.click();
            await new Promise((resolve,reject)=>{
              const timeout=setTimeout(()=>reject(Error('playback stalled')),6000);
              function poll(){if(Number(slider.value)>=20){clearTimeout(timeout);resolve()}else requestAnimationFrame(poll)}
              poll();
            });
            const elapsed=performance.now()-start;
            button.click();
            const pausedAt=slider.value;
            await new Promise(resolve=>setTimeout(resolve,Math.max(100,interval*2)));
            return {interval_ms:interval, first_20_frames_ms:Math.round(elapsed),
                    pause_stable:slider.value===pausedAt};
          }
          return {default_interval_ms:defaultInterval,
                  previous_default:await run(180), current_default:await run(defaultInterval)};
        }""")
        report = {"schema": "bounded-browser-replay-latency-v1", "html": str(args.html.resolve()),
                  "startup_ms": round(startup_ms, 2),
                  "frame_change": summarize(profile.pop("changes")),
                  "playback": playback, **profile}
        browser.close()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
