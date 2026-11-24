/* EchoAlign project page — interactive figures.
   All numbers come from window.ECHO (data.js), extracted from the manuscript's tables and figures. */
(() => {
"use strict";
const E = window.ECHO;
const MODELS = E.models; // EchoAlign, EchoPrime, PanEcho, EchoCLIP, EchoJEPA, EchoFM
const MC = m => `var(--m${typeof m === "number" ? m : MODELS.indexOf(m)})`;
const STRUCT = ["LV", "LA", "RV", "RA", "AV", "MV", "TV", "PV", "AoR", "Arch", "PA", "Peri"];
const SNAME = { LV: "Left ventricle", LA: "Left atrium", RV: "Right ventricle", RA: "Right atrium", AV: "Aortic valve", MV: "Mitral valve", TV: "Tricuspid valve", PV: "Pulmonic valve", AoR: "Aortic root", Arch: "Aortic arch", PA: "Pulmonary artery", Peri: "Pericardium", Other: "Other / global" };
const SC = s => s === "Other" ? "var(--m3)" : `var(--s-${s})`;
const GRADES = ["Normal", "Trace or borderline", "Mild", "Moderate", "Severe"];
const GC = i => `var(--g${i})`;
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const DUR = reduceMotion ? 0 : 750;
const fmt3 = d3.format(".3f"), fmt2 = d3.format(".2f");
const $ = s => document.querySelector(s);

/* example report: synthetic. Written for this page in generic wording; it does not reproduce any
   patient report from the private (iCardio.ai) dataset or the paper's report figure. */
const REPORT = {
  LV: "Left ventricle normal in size with preserved systolic function. Estimated EF 55-60%.",
  LA: "Left atrium not dilated.",
  RV: "Right ventricle normal in size and function.",
  RA: "Right atrium not dilated.",
  AV: "Three cusps with mild sclerosis; no stenosis.",
  MV: "Leaflets open well. Mild mitral regurgitation.",
  TV: "Leaflets appear normal. Mild tricuspid regurgitation.",
  PV: "Not well seen; no significant regurgitation.",
  AoR: "Within normal limits.",
  Arch: "",
  PA: "Estimated pulmonary pressure within normal limits.",
  Peri: "No pericardial effusion."
};
const CONCLUSION = "Preserved biventricular function. Mild mitral and tricuspid regurgitation.";

/* ---------- utilities ---------- */
const tip = $("#tip");
function showTip(html, ev) {
  tip.innerHTML = html; tip.classList.add("on");
  const pad = 14, r = tip.getBoundingClientRect();
  let x = ev.clientX + pad, y = ev.clientY + pad;
  if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - pad;
  if (y + r.height > innerHeight - 8) y = ev.clientY - r.height - pad;
  tip.style.left = Math.max(8, x) + "px"; tip.style.top = Math.max(8, y) + "px";
}
const hideTip = () => tip.classList.remove("on");
function modelTable(rows, digits = 3) {
  // rows: [[model, value, extra?]] sorted desc
  return `<table>${rows.map(([m, v, ex]) => `<tr class="${m === "EchoAlign" ? "ea" : ""}"><td><span class="dot" style="background:${MC(m)}"></span>${m}${ex || ""}</td><td>${v == null ? "–" : v.toFixed(digits)}</td></tr>`).join("")}</table>`;
}
function onVisible(el, fn, margin = "0px 0px -15% 0px") {
  if (!el) return;
  const io = new IntersectionObserver(es => { es.forEach(e => { if (e.isIntersecting) { io.disconnect(); fn(); } }); }, { rootMargin: margin });
  io.observe(el);
}
function segmented(id, cb) {
  const el = document.getElementById(id);
  el.addEventListener("click", e => {
    const b = e.target.closest("button"); if (!b) return;
    el.querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b));
    cb(b.dataset.v);
  });
  return () => el.querySelector("button.on")?.dataset.v;
}
function legendModels(el, models = MODELS, extra = {}) {
  el.innerHTML = models.map(m => `<span class="k ${m === "EchoAlign" ? "ea" : ""}"><span class="sw" style="background:${MC(m)}"></span>${m}${extra[m] || ""}</span>`).join("");
}
function width(el) { return Math.max(280, el.getBoundingClientRect().width); }
const isBig = el => !!(el && el.closest && el.closest(".modal-body"));
function onResize(fn) { let t; let w = innerWidth; addEventListener("resize", () => { clearTimeout(t); t = setTimeout(() => { if (innerWidth !== w) { w = innerWidth; fn(); } }, 180); }); addEventListener("redraw", () => fn()); }
const themeHooks = [];
function cssVar(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }
function logit(p) { p = Math.min(0.998, Math.max(0.3, p)); return Math.log(p / (1 - p)); }

/* ---------- theme ---------- */
(function theme() {
  const root = document.documentElement;
  try { const t = localStorage.getItem("ea-theme"); if (t) root.dataset.theme = t; } catch (e) {}
  $("#themebtn").addEventListener("click", () => {
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("ea-theme", root.dataset.theme); } catch (e) {}
    themeHooks.forEach(f => f());
  });
})();

const redraw = () => setTimeout(() => window.dispatchEvent(new Event("redraw")), 30);

/* ---------- beating logo: tag the four chambers (RV, LV, RA, LA) ---------- */
function tagChambers(svg) {
  const ps = svg.querySelectorAll("g[clip-path] > path");
  ["ch-v", "ch-v", "ch-a", "ch-a"].forEach((c, i) => ps[i]?.classList.add(c));
}
document.querySelectorAll("svg.mark").forEach(tagChambers);
document.querySelector(".brand svg.mark")?.classList.add("pump");
let pumpN = 0;
function pumpLogo() {
  const id = "lgP" + (++pumpN);
  return document.querySelector(".brand svg.mark").outerHTML.replaceAll('id="lgA"', `id="${id}"`).replaceAll("url(#lgA)", `url(#${id})`);
}

const BEAT = 850;    // ms per heartbeat (~70 bpm), shared by the logo's beating chambers and the intro timing
document.documentElement.style.setProperty("--beat", BEAT + "ms");
/* ---------- waiting logo: echo view -> structures -> leads -> report sentences, looping ---------- */
function alignLoader(host, opts = {}) {
  // timing runs on the heartbeat: one step per beat (BEAT matches --beat in styles.css)
  const k = (opts.speed || 1) * BEAT, loops = opts.loops || Infinity; let done = 0, stopped = false;
  host.innerHTML = `<div class="al-logo">${pumpLogo()}</div><div class="al-cap"><span></span></div>`;
  const svg = host.querySelector("svg"), cap = host.querySelector(".al-cap span");
  const fan = svg.querySelector(":scope > path");
  const ch = [...svg.querySelectorAll("g[clip-path] > path")];            // RV, LV, RA, LA
  const leads = [...svg.querySelectorAll("g[stroke-dasharray] > path")];  // RV, RA (left) · LV, LA (right)
  const bars = [...svg.querySelectorAll(":scope > rect")];                // RV, RA (left) · LV, LA (right)
  // pair each chamber with its lead and sentence bar
  const pairs = [[0, 0, 0, true], [1, 2, 2, false], [2, 1, 1, true], [3, 3, 3, false]].map(([c, l, b, left]) => ({ c: ch[c], l: leads[l], b: bars[b], left }));
  const NAMES = ["Right ventricle", "Left ventricle", "Right atrium", "Left atrium"];
  const ease = "cubic-bezier(.2,.8,.2,1)";
  const anim = (el, kf, delay, dur) => el.animate(kf, { delay: delay * k, duration: dur * k, easing: ease, fill: "both", id: "al" });
  const say = (txt, delay) => setTimeout(() => { if (!host.isConnected || stopped) return; cap.style.opacity = 0; setTimeout(() => { cap.textContent = txt; cap.style.opacity = 1; }, 160); }, delay * k);
  let timers = [];
  function cycle() {
    if (!host.isConnected || stopped) return;
    svg.getAnimations({ subtree: true }).filter(a => a.id === "al").forEach(a => a.cancel());
    timers.forEach(clearTimeout); timers = [];
    if (reduceMotion) { cap.textContent = "Each cardiac structure aligned with its own report sentence"; return; }
    if (!opts.keepFan) anim(fan, [{ opacity: 0, transform: "scale(.92)" }, { opacity: 1, transform: "scale(1)" }], 0, .45);
    fan.style.transformOrigin = "48px 4px";
    timers.push(say("1 · the echo view", 0));
    pairs.forEach((p, i) => anim(p.c, [{ opacity: 0 }, { opacity: 1 }], 1.6 + i * .12, .3));
    timers.push(say("2 · the cardiac structures it shows", 1.6));
    pairs.forEach((p, i) => {
      const t = 2.8 + i;   // one structure per beat
      anim(p.l, [{ clipPath: p.left ? "inset(0 0 0 100%)" : "inset(0 100% 0 0)" }, { clipPath: "inset(0 0 0 0)" }], t, .4);
      p.b.style.transformBox = "fill-box"; p.b.style.transformOrigin = p.left ? "100% 50%" : "0% 50%";
      anim(p.b, [{ transform: "scaleX(0)", opacity: 0 }, { transform: "scaleX(1)", opacity: 1 }], t + .38, .36);
      timers.push(say(`3 · ${NAMES[i]} → its report sentence`, t));
    });
    timers.push(say("Each structure aligned with its own report text", 6.8));
    if (++done >= loops) { timers.push(setTimeout(() => !stopped && opts.onDone && opts.onDone(), 9.8 * k)); return; }   // last loop: stay drawn
    const all = [fan, ...ch, ...leads, ...bars];
    all.forEach(el => anim(el, [{ opacity: 1 }, { opacity: 0 }], 10.3, .45));
    timers.push(setTimeout(cycle, 10.9 * k));
  }
  // hover a chamber or its sentence: highlight that pair
  pairs.forEach((p, i) => [p.c, p.b, p.l].forEach(el => {
    el.style.cursor = "pointer";
    el.addEventListener("pointerenter", () => { pairs.forEach((q, j) => [q.c, q.l, q.b].forEach(e => e.style.filter = j === i ? "brightness(1.15) drop-shadow(0 0 2px rgba(0,0,0,.25))" : "opacity(.28)")); cap.textContent = `${NAMES[i]} ↔ its report sentence`; });
    el.addEventListener("pointerleave", () => pairs.forEach(q => [q.c, q.l, q.b].forEach(e => e.style.filter = "")));
  }));
  if (!opts.defer) cycle();
  return {
    svg, cap, start: cycle,
    clear(ms) { [...ch, ...leads, ...bars].forEach(el => el.animate([{ opacity: 1 }, { opacity: 0 }], { duration: ms, easing: ease, fill: "both", id: "al" })); },
    stop() { stopped = true; timers.forEach(clearTimeout); svg.getAnimations({ subtree: true }).filter(a => a.id === "al").forEach(a => a.finish()); }
  };
}

/* ---------- opening intro: big logo plays the alignment twice, then settles into the title block ---------- */
(function intro() {
  const text = document.querySelector("#overview .hero-text"), mark = text && text.querySelector(".hero-mark .mark");
  const pending = () => document.documentElement.classList.remove("intro-pending");
  const replayBtn = $("#replayIntro");
  if (!text || !mark) { pending(); return; }
  if (reduceMotion && replayBtn) replayBtn.remove();
  const kids = [...text.children].filter(el => !el.classList.contains("intro-ov"));
  const hm = text.querySelector(".hero-mark");
  const EASE = "cubic-bezier(.65,0,.25,1)";
  let running = false;
  // fromSmall: replay — the small mark zooms out to full size first, then the sequence plays
  function run(fromSmall) {
    if (running) return; running = true;
    const ov = document.createElement("div"); ov.className = "intro-ov"; ov.innerHTML = `<div class="pump-logo"></div><button class="intro-skip" type="button">Skip intro</button>`;
    if (fromSmall) hm.style.transition = "none";   // the big copy starts exactly where the small mark is
    text.classList.add("intro"); text.appendChild(ov); pending();
    const ld = alignLoader(ov.querySelector(".pump-logo"), { loops: 1, onDone: settle, defer: true, keepFan: fromSmall });
    const big = ld.svg; big.style.transformOrigin = "0 0";
    if (!fromSmall) {   // first visit: wait until the page has finished its start-up work so every step gets its full beat
      [big.querySelector(":scope > path"), ...big.querySelectorAll("g[clip-path] > path, g[stroke-dasharray] > path, :scope > rect")].forEach(el => el.style.opacity = 0);
      const go = () => setTimeout(() => { if (settled) return; big.querySelectorAll("path, rect").forEach(el => el.style.opacity = ""); ld.start(); }, 250);
      document.readyState === "complete" ? go() : addEventListener("load", go, { once: true });
    }
    const toMark = () => { const a = big.getBoundingClientRect(), b = mark.getBoundingClientRect(); return `translate(${b.left - a.left}px, ${b.top - a.top}px) scale(${b.width / a.width})`; };
    if (fromSmall) {
      ld.cap.style.opacity = 0;
      const zin = big.animate([{ transform: toMark() }, { transform: "none" }], { duration: 900, easing: EASE });
      zin.onfinish = () => { if (settled) return; ld.cap.style.opacity = ""; ld.clear(350); setTimeout(() => !settled && ld.start(), 380); };
    }
    let settled = false;
    function settle() {
      if (settled) return; settled = true;
      ld.stop();
      ld.cap.style.opacity = 0; ov.querySelector(".intro-skip").style.opacity = 0;
      big.getAnimations().forEach(a => a.cancel());
      const fly = big.animate([{ transform: "none" }, { transform: toMark() }], { duration: 950, easing: EASE, fill: "forwards" });
      fly.onfinish = () => {
        hm.style.transition = "none";   // the small mark takes over exactly where the big one landed
        text.classList.remove("intro"); text.classList.add("intro-out");
        requestAnimationFrame(() => requestAnimationFrame(() => hm.style.transition = ""));
        kids.forEach((el, i) => el.style.transitionDelay = `${i === 0 ? 0 : 0.08 + i * 0.11}s`);
        ov.remove(); running = false;
        setTimeout(() => { text.classList.remove("intro-out"); kids.forEach(el => el.style.transitionDelay = ""); }, 2200);
      };
    }
    ov.querySelector(".intro-skip").addEventListener("click", settle);
    const skip = () => settle();
    addEventListener("wheel", skip, { once: true, passive: true });
    addEventListener("touchmove", skip, { once: true, passive: true });
    addEventListener("keydown", skip, { once: true });
  }
  replayBtn && replayBtn.addEventListener("click", () => { if (scrollY > 40) $("#overview").scrollIntoView(); run(true); });
  if (reduceMotion || scrollY > 40 || location.hash && location.hash !== "#overview") { pending(); return; }
  run(false);
})();

/* ---------- section nav, dots, copy ---------- */
(function nav() {
  const slides = [...document.querySelectorAll(".slide")];
  const dots = $("#dots");
  dots.innerHTML = slides.map(s => `<a href="#${s.id}" title="${s.dataset.name}"></a>`).join("");
  const links = [...document.querySelectorAll("#navlinks a")];
  const io = new IntersectionObserver(es => es.forEach(e => {
    if (!e.isIntersecting) return;
    const id = e.target.id;
    dots.querySelectorAll("a").forEach(a => a.classList.toggle("on", a.getAttribute("href") === "#" + id));
    links.forEach(a => a.classList.toggle("active", a.getAttribute("href") === "#" + id));
  }), { threshold: 0.45 });
  slides.forEach(s => io.observe(s));
  document.querySelectorAll(".copy").forEach(b => b.addEventListener("click", () => {
    const pre = document.getElementById(b.dataset.copy);
    const txt = [...pre.childNodes].filter(n => n !== b).map(n => n.textContent).join("");
    navigator.clipboard?.writeText(txt).then(() => { b.textContent = "Copied"; setTimeout(() => b.textContent = "Copy", 1400); });
  }));
  addEventListener("keydown", e => {
    if (e.target.closest("input,select,textarea") || $("#modal").classList.contains("on")) return;
    const i = slides.findIndex(s => { const r = s.getBoundingClientRect(); return r.top > -innerHeight / 2 && r.top < innerHeight / 2; });
    if (e.key === "PageDown" || (e.key === "ArrowDown" && e.altKey)) { e.preventDefault(); slides[Math.min(slides.length - 1, i + 1)]?.scrollIntoView(); }
    if (e.key === "PageUp" || (e.key === "ArrowUp" && e.altKey)) { e.preventDefault(); slides[Math.max(0, i - 1)]?.scrollIntoView(); }
  });
})();

/* ---------- zoom modal ---------- */
const Zoom = (function () {
  const modal = $("#modal"), body = $("#modalBody"), title = $("#modalTitle");
  let cur = null, ph = null, onClose = null;
  function open(card) {
    close();
    cur = card; ph = document.createElement("div"); ph.className = "ph";
    ph.style.height = card.offsetHeight + "px"; ph.style.display = "block";
    card.parentNode.insertBefore(ph, card);
    body.appendChild(card); title.textContent = card.dataset.title || "";
    modal.classList.add("on"); modal.setAttribute("aria-hidden", "false"); redraw();
  }
  function openCustom(t, build, closeCb) {
    close(); title.textContent = t; body.innerHTML = ""; const d = document.createElement("div"); d.style.height = "100%"; body.appendChild(d);
    modal.classList.add("on"); modal.setAttribute("aria-hidden", "false"); onClose = closeCb; build(d);
  }
  function close() {
    if (!modal.classList.contains("on")) return;
    if (cur) { ph.parentNode.insertBefore(cur, ph); ph.remove(); cur = null; ph = null; }
    else body.innerHTML = "";
    if (onClose) { onClose(); onClose = null; }
    modal.classList.remove("on"); modal.setAttribute("aria-hidden", "true"); redraw();
  }
  $("#modalClose").addEventListener("click", close);
  modal.addEventListener("click", e => { if (e.target === modal) close(); });
  addEventListener("keydown", e => { if (e.key === "Escape") close(); });
  document.querySelectorAll(".card.zoomable").forEach(card => {
    const b = document.createElement("button"); b.className = "zoombtn"; b.title = "Zoom"; b.setAttribute("aria-label", "Zoom " + (card.dataset.title || ""));
    b.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/></svg>';
    b.addEventListener("click", e => { e.stopPropagation(); open(card); });
    card.appendChild(b);
  });
  return { open, openCustom, close };
})();

/* ---------- results tabs ---------- */
(function tabs() {
  const t = $("#rtabs");
  t.addEventListener("click", e => {
    const b = e.target.closest("button"); if (!b) return;
    t.querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b));
    document.querySelectorAll("#results .tabpane").forEach(p => p.classList.toggle("on", p.dataset.t === b.dataset.t));
    redraw();
  });
})();

/* ---------- visibility helpers ---------- */
const VIS = E.visibility.rows;
const visOf = view => { const r = VIS.find(v => v.view === view); return Object.fromEntries(STRUCT.map((s, i) => [s, r.w[i]])); };
const whenHeart = fn => { if (window.Heart3D) fn(); else addEventListener("heart3d-ready", fn, { once: true }); };

/* =========================================================
   CAMUS cine with structure queries (real clip, expert-mask positions)
   ========================================================= */
const CAM = window.CAMUS;
function makeCine(host, opts = {}) {
  const st = { view: opts.view || "A4C", mode: opts.mode || "rings", focus: null, rate: 1 };
  host.innerHTML = `<div class="cine"><video muted loop playsinline autoplay preload="auto"></video><svg></svg></div>` +
    (opts.controls === false ? "" : `<div class="cine-ctl"><div class="seg cv"><button data-v="A4C">A4C</button><button data-v="A2C">A2C</button></div>
      <div class="seg cr"><button data-v="1" class="on">1×</button><button data-v=".5">½×</button><button data-v="0">❚❚</button></div>
      <span class="src">CAMUS patient0310 · solid = expert mask · dashed = placed by eye</span></div>`);
  const video = host.querySelector("video"), svg = d3.select(host.querySelector("svg")), cineEl = host.querySelector(".cine");
  let live = {}, last = -1, extras = null;
  const V = () => CAM.views[st.view];
  function load() {
    const ext = video.canPlayType('video/webm; codecs="vp9"') ? "webm" : "mp4";
    video.src = `assets/camus/camus_patient0310_${st.view}_loop.${ext}`;
    video.addEventListener("loadedmetadata", () => { video.playbackRate = st.rate || 1; if (st.rate) video.play().catch(() => {}); }, { once: true });
    build();
    host.querySelectorAll(".cv button").forEach(b => b.classList.toggle("on", b.dataset.v === st.view));
  }
  let pad = { l: 0, r: 0, t: 0, b: 0 }, fk = 1;
  function build() {
    const v = V(); live = {}; last = -1;
    // rings / leaders: keep the scan inside a margin; the query legend and captions get their own space
    const framed = st.mode === "rings" || st.mode === "leaders", square = st.mode === "patches";
    const px0 = (host.clientWidth || 900) / (v.W + 290);                   // approx. screen px per video unit
    fk = framed ? Math.min(2.1, Math.max(1, 10.5 / (26 * px0))) : 1;        // overlay text stays >= ~10.5 px
    const SQ = Math.max(v.W, v.H);
    pad = square ? { l: (SQ - v.W) / 2, r: (SQ - v.W) / 2, t: (SQ - v.H) / 2, b: (SQ - v.H) / 2 } : framed ? (opts.queriesLeft ? { l: 250, r: 40, t: 34, b: 96 * fk } : { l: 40, r: 250, t: 34, b: 96 * fk }) : { l: 0, r: 0, t: 0, b: 0 };
    const TW = v.W + pad.l + pad.r, TH = v.H + pad.t + pad.b;
    svg.attr("viewBox", `${-pad.l} ${-pad.t} ${TW} ${TH}`).selectAll("*").remove();
    if (framed || square) {
      cineEl.style.aspectRatio = `${TW} / ${TH}`;
      Object.assign(video.style, { position: "absolute", left: `${100 * pad.l / TW}%`, top: `${100 * pad.t / TH}%`, width: `${100 * v.W / TW}%`, height: `${100 * v.H / TH}%` });
    }
    const g = svg.append("g");
    extras = svg.append("g");
    if (st.mode === "plain") { drawFoot(); return; }
    if (st.mode === "patches") { drawPatches(); return; }
    const LX = v.W + pad.r - 70, LY0 = fk > 1.2 ? 230 : 150, LDY = Math.max(52, 24 * fk * 1.32), QX = opts.queriesLeft ? -pad.l + 64 : LX;
    const leaders = g.append("g").attr("pointer-events", "none"), rings = g.append("g"), legend = g.append("g");
    legend.append("text").attr("x", QX + (opts.queriesLeft ? -34 : 36)).attr("y", LY0 - 48).attr("text-anchor", opts.queriesLeft ? "start" : "end").attr("fill", "#e6e9ed").attr("font-size", 26 * Math.min(fk, 1.8)).attr("font-family", "var(--mono)").text("queries");
    STRUCT.forEach((k, i) => {
      const w = v.w[k], y = LY0 + i * LDY, on = w > 0, located = on && CAM.pos[st.view][0][k] !== undefined, byEye = (v.anchored || {})[k] !== undefined;
      const row = legend.append("g").style("cursor", "pointer");
      const tx = opts.queriesLeft ? QX + 42 : QX - 14;
      row.append("text").attr("x", tx).attr("y", y + 9).attr("text-anchor", opts.queriesLeft ? "start" : "end").attr("fill", located ? "#e6e9ed" : "#8b949e").attr("font-size", 24 * fk).attr("font-family", "var(--mono)").text(k);
      const cx = opts.queriesLeft ? QX - 34 : QX;
      if (on && !located) row.append("rect").attr("x", cx + 1).attr("y", y - 14).attr("width", 32).attr("height", 28).attr("rx", 5).attr("fill", SCX(k)).attr("fill-opacity", .22).attr("stroke", SCX(k)).attr("stroke-width", 2.5).attr("stroke-dasharray", "5 4");
      else row.append("rect").attr("x", cx).attr("y", y - 15).attr("width", 34).attr("height", 30).attr("rx", 5).attr("fill", on ? SCX(k) : "#2a2d31").attr("stroke", on ? "none" : "#3a3e44");
      row.append("title").text(`${SNAME[k]} · weight ${w}${!on ? " (not expected in this view)" : !located ? " (not seen on this clip)" : byEye ? " · placed by eye" : " · expert mask"}`);
      if (!located) return;
      const r = 10 + 32 * w;
      const path = leaders.append("path").attr("fill", "none").attr("stroke", SCX(k)).attr("stroke-width", 2.5).attr("stroke-dasharray", "7 7").attr("class", "lead");
      const rg = rings.append("g").style("cursor", "pointer");
      const ring = rg.append("circle").attr("r", r).attr("fill", SCX(k)).attr("fill-opacity", .18).attr("stroke", SCX(k)).attr("stroke-width", 4).attr("stroke-dasharray", byEye ? "9 6" : null);
      const dot = rg.append("circle").attr("r", 7).attr("fill", "#fff");
      live[k] = { path, ring, dot, r, y, rg, cx: opts.queriesLeft ? cx + 34 : cx - 4 };
      const show = on2 => { if (st.mode !== "leaders") path.style("opacity", on2 ? .9 : 0); ring.attr("fill-opacity", on2 ? .34 : .18); };
      [rg, row].forEach(n => n.on("pointerenter", () => show(true)).on("pointerleave", () => show(false)));
      path.style("opacity", st.mode === "leaders" ? .85 : 0);
    });
    applyFocus(); drawFoot(); draw(frameAt(video.currentTime || 0));
  }
  function drawFoot() {
    const v = V();
    // skip the caption where it would render below ~9 px (small tiles); the tile title already names the view
    const px = (host.clientWidth || 600) / (v.W + pad.l + pad.r);
    if (26 * fk * px < 9) return;
    svg.append("text").attr("x", -pad.l + 36).attr("y", v.H + pad.b - 30).attr("fill", "#e6e9ed").attr("font-size", 26 * fk).attr("font-family", "var(--mono)").text(`${st.view} · ${v.title}`);
    const n = STRUCT.filter(k => v.w[k] > 0).length;
    // (the legend already shows which structures have weight > 0)
  }
  function drawPatches() {
    if (pad.t > 0) extras.append("text").attr("x", V().W / 2).attr("y", -pad.t / 2 + 12).attr("text-anchor", "middle").attr("fill", "#8b949e").attr("font-size", Math.max(V().W, V().H) / 34).attr("font-family", "var(--mono)").text("padded to a square · 14 × 14 patches");
    // 14 x 14 patch grid on the 224 x 224 square input (the sector is padded to a square)
    const v = V(), S = Math.max(v.W, v.H), x0 = -pad.l, y0 = -pad.t, c = S / 14;
    const g = extras.append("g");
    for (let i = 0; i < 14; i++) for (let j = 0; j < 14; j++)
      g.append("rect").attr("x", x0 + j * c).attr("y", y0 + i * c).attr("width", c - 2).attr("height", c - 2).attr("fill", "#9cc3ff").attr("fill-opacity", 0).attr("stroke", "#9cc3ff").attr("stroke-opacity", .35).attr("stroke-width", 1.2)
        .transition().delay(reduceMotion ? 0 : (i + j) * 45).duration(400).attr("fill-opacity", .16).transition().duration(600).attr("fill-opacity", .04);
  }
  function applyFocus() {
    for (const k in live) live[k].rg.style("opacity", !st.focus || st.focus === k ? 1 : .2);
  }
  function draw(f) {
    const P = CAM.pos[st.view][f]; if (!P) return;
    for (const k in live) {
      const L = live[k], [x, y] = P[k];
      L.ring.attr("cx", x).attr("cy", y); L.dot.attr("cx", x).attr("cy", y);
      const sx = x + (opts.queriesLeft ? -L.r * .7 : L.r * .7), sy = y - L.r * .7, ex = L.cx, ey = L.y;
      const k1 = opts.queriesLeft ? -180 : 180;
      L.path.attr("d", `M${sx},${sy} C${sx + k1},${sy - 60} ${ex - k1 * 1.2},${ey} ${ex},${ey}`);
    }
  }
  function frameAt(t) { const v = V(), L = 2 * v.n - 2, i = Math.floor(t * v.fps + 0.5) % L; return i < v.n ? i : L - i; }
  let off = 0;
  function tick(now, meta) {
    const t = meta && meta.mediaTime !== undefined ? meta.mediaTime : video.currentTime;
    const f = frameAt(t); if (f !== last) { draw(f); last = f; }
    if (st.mode === "leaders" && !reduceMotion) { off -= 1.2; svg.selectAll(".lead").attr("stroke-dashoffset", off); }
    schedule();
  }
  function schedule() { if (video.requestVideoFrameCallback && !video.paused) video.requestVideoFrameCallback(tick); else requestAnimationFrame(() => tick()); }
  host.querySelectorAll(".cv button").forEach(b => b.addEventListener("click", () => { st.view = b.dataset.v; load(); opts.onView && opts.onView(st.view); }));
  host.querySelectorAll(".cr button").forEach(b => b.addEventListener("click", () => {
    host.querySelectorAll(".cr button").forEach(x => x.classList.toggle("on", x === b));
    st.rate = +b.dataset.v; if (st.rate) { video.playbackRate = st.rate; video.play().catch(() => {}); } else video.pause();
  }));
  load(); schedule();
  return { setView(v) { st.view = v; load(); }, setMode(m) { st.mode = m; build(); }, focus(k) { st.focus = k; applyFocus(); }, get view() { return st.view; }, video };
}
const SCX = k => ({ LV: "#B8322A", LA: "#E0735A", RV: "#2C5F9E", RA: "#6E9BD1", AV: "#B7791F", MV: "#DDAA3C", TV: "#2F8062", PV: "#76B99A", AoR: "#5E3A96", Arch: "#9D82C6", PA: "#2A8A9A", Peri: "#7C7A72" })[k];

(function hero() {
  makeCine($("#cineHero"), { view: "A4C", mode: "rings" });
  onVisible($("#herostats"), () => {
    document.querySelectorAll("#herostats .n").forEach((el, i) => {
      const to = +el.dataset.count, dec = +(el.dataset.dec || 0), suf = el.dataset.suf || "";
      const f = d3.format(dec ? `.${dec}f` : ",d");
      d3.select(el).transition().delay(i * 90).duration(reduceMotion ? 0 : 1300).ease(d3.easeCubicOut).tween("t", () => t => { el.textContent = f(to * t) + suf; });
    });
  }, "0px");
})();

/* =========================================================
   PROBLEM: 3D plane, anatomical slice, report supervision
   ========================================================= */
function drawSlice(el, viewName, opts = {}) {
  const V = window.VIEWS3D[viewName], w = visOf(viewName);
  el.innerHTML = "";
  const a = V.ang * Math.PI / 180, D = V.depth, half = D * Math.sin(a) + 8;
  // screen pixels per viewBox unit, so labels keep a readable size at any tile size
  const pxu = Math.min((el.clientWidth || 320) / (2 * half), (el.clientHeight || el.clientWidth || 320) / (D + 22));
  const svg = d3.select(el).append("svg").attr("viewBox", `${-half} -10 ${2 * half} ${D + 22}`).attr("preserveAspectRatio", "xMidYMid meet");
  const id = "clip" + Math.random().toString(36).slice(2, 8);
  const fan = `M0,0 L${D * Math.sin(-a)},${D * Math.cos(a)} A${D},${D} 0 0 0 ${D * Math.sin(a)},${D * Math.cos(a)} Z`;
  svg.append("clipPath").attr("id", id).append("path").attr("d", fan);
  const grad = svg.append("radialGradient").attr("id", id + "g").attr("cx", "50%").attr("cy", "0%").attr("r", "100%");
  grad.append("stop").attr("offset", "0%").attr("stop-color", "#2a2f36"); grad.append("stop").attr("offset", "100%").attr("stop-color", "#101318");
  svg.append("path").attr("d", fan).attr("fill", `url(#${id}g)`);
  const g = svg.append("g").attr("clip-path", `url(#${id})`);
  const order = ["Peri", "PVn", "SVC", "IVC", "LA", "RA", "LV", "RV", "IVS", "PAP", "AoR", "Arch", "PA", "MV", "TV", "AV", "PV"];
  const ctx = { IVS: "#b7a197", PAP: "#a88b80", SVC: "#8f969e", IVC: "#8f969e", PVn: "#a3a9b0" };
  const line = d3.line().curve(d3.curveCatmullRom.alpha(.5));
  const labels = [];
  order.forEach(k => {
    const loops = V.sections[k]; if (!loops) return;
    const isS = STRUCT.includes(k), wk = isS ? w[k] : null;
    const col = isS ? (wk > 0 ? SCX(k) : "#6b6e73") : ctx[k];
    const fillOp = isS ? (wk > 0 ? .3 + .45 * wk : .12) : .22;
    loops.forEach(L => {
      const p = g.append("path").attr("d", line(L.p) + (L.c ? "Z" : ""))
        .attr("fill", L.c && k !== "Peri" ? col : "none").attr("fill-opacity", fillOp)
        .attr("stroke", col).attr("stroke-width", k === "Peri" ? .9 : 1.1).attr("stroke-opacity", isS && wk === 0 ? .45 : .95)
        .attr("stroke-dasharray", k === "Peri" ? "3 2" : null).style("cursor", "pointer")
        .on("mousemove", ev => showTip(`<div class="tt">${(window.HEART_NAMES || SNAME)[k] || SNAME[k]}</div>${isS ? `Visibility weight in ${viewName}: <b>${wk}</b>` : "Context structure"}`, ev))
        .on("mouseleave", hideTip);
      if (!reduceMotion && opts.animate) { const len = p.node().getTotalLength(); p.attr("stroke-dasharray", `${len} ${len}`).attr("stroke-dashoffset", len).transition().duration(900).attr("stroke-dashoffset", 0).on("end", () => p.attr("stroke-dasharray", k === "Peri" ? "3 2" : null)); }
    });
    if (isS && wk > 0 && k !== "Peri") {
      const big = loops.reduce((a2, b2) => b2.p.length > a2.p.length ? b2 : a2);
      const cx = d3.mean(big.p, d => d[0]), cy = d3.mean(big.p, d => d[1]);
      labels.push([k, cx, cy]);
    }
  });
  labels.forEach(([k, x, y]) => g.append("text").attr("x", x).attr("y", y).attr("text-anchor", "middle").attr("dy", "0.35em")
    .attr("fill", "#fff").attr("font-size", Math.max(D / 26, 10.5 / pxu)).attr("font-weight", 700).attr("font-family", "var(--sans)").attr("paint-order", "stroke").attr("stroke", "#000").attr("stroke-width", D / 120).attr("pointer-events", "none").text(k));
  svg.append("path").attr("d", fan).attr("fill", "none").attr("stroke", "#3a4049");
  // depth marks only when they can be read; the view name is already in the card or tile title
  if ((el.clientWidth || 320) >= 300) for (let d = 50; d < D - 25; d += 50) svg.append("text").attr("x", D * Math.sin(a) * d / D + 4).attr("y", D * Math.cos(a) * d / D).attr("fill", "#8b949e").attr("font-size", Math.max(D / 34, 9.5 / pxu)).attr("font-family", "var(--mono)").text(d / 10 + " cm");
}

(function problem() {
  const names = Object.keys(window.VIEWS3D);
  let view = "A4C", mode = "full", touched = false, H = null;
  $("#viewchips").innerHTML = names.map(n => `<button class="chip ${n === view ? "on" : ""}" data-v="${n}">${window.VIEWS3D[n].label}</button>`).join("");
  $("#viewchips").addEventListener("click", e => { const b = e.target.closest(".chip"); if (!b) return; view = b.dataset.v; $("#viewchips").querySelectorAll(".chip").forEach(c => c.classList.toggle("on", c === b)); render(true); });
  const setMode = v => { mode = v; document.querySelectorAll("#pairmode button").forEach(b => b.classList.toggle("on", b.dataset.v === v)); renderReport(); };
  segmented("pairmode", v => { touched = true; setMode(v); });
  $("#cutToggle").addEventListener("change", e => { H && H.setCut(e.target.checked); if (!e.target.checked && H) H.resetCamera(); });
  const rep = $("#report");
  rep.innerHTML = STRUCT.map(s => `<div class="row" data-s="${s}"><div class="sname"><i style="background:${SC(s)}"></i>${SNAME[s]}</div><div class="txt">${REPORT[s] || "<i>(no finding)</i>"}</div><div class="tag"></div></div>`).join("");
  rep.addEventListener("mouseover", e => { const r = e.target.closest(".row"); if (r && H) H.highlight(r.dataset.s); });
  rep.addEventListener("mouseleave", () => H && H.highlight(null));
  function renderReport() {
    const w = visOf(view); let bad = 0, nonEmpty = 0;
    rep.querySelectorAll(".row").forEach(r => {
      const s = r.dataset.s, ws = w[s], empty = !REPORT[s], tag = r.querySelector(".tag");
      r.className = "row"; r.style.opacity = "";
      if (empty) { r.classList.add("off"); tag.textContent = "empty"; return; }
      nonEmpty++;
      if (mode === "full") { if (ws === 0) { r.classList.add("bad"); tag.textContent = "not visible"; bad++; } else { r.classList.add("ok"); tag.textContent = "target"; } }
      else if (ws === 0) { r.classList.add("off"); tag.textContent = "masked"; bad++; }
      else { r.classList.add("ok"); tag.textContent = `w = ${ws}`; r.style.opacity = .45 + .55 * ws; }
    });
    const good = nonEmpty - bad, label = window.VIEWS3D[view].label;
    $("#pmeter").innerHTML = `<div class="meter"><div style="width:${100 * good / nonEmpty}%;background:var(--s-TV)"></div><div style="width:${100 * bad / nonEmpty}%;background:${mode === "full" ? "var(--accent)" : "var(--rule)"}"></div></div>
      <div class="note">${mode === "full" ? `<b style="color:var(--accent)">${bad} of ${nonEmpty}</b> findings describe structures the ${label} view does not show, yet all supervise this video.`
        : `<b>${good} of ${nonEmpty}</b> findings supervise this video, each weighted by its visibility; ${bad} masked.`}</div>`;
  }
  function render(anim) {
    $("#sliceTitle").textContent = window.VIEWS3D[view].label;
    drawSlice($("#sliceProblem"), view, { animate: anim });
    renderReport();
    if (H) { H.setWeights(visOf(view)); H.setView(view); }
  }
  render(false);
  onVisible($("#heartProblem"), () => whenHeart(() => { H = new Heart3D($("#heartProblem")); H.setWeights(visOf(view)); H.setView(view); }), "200px");
  onVisible(rep, () => setTimeout(() => { if (!touched) setMode("ea"); }, 2800), "0px 0px -30% 0px");
  window.addEventListener("redraw", () => drawSlice($("#sliceProblem"), view));
})();

/* =========================================================
   METHOD: step player
   ========================================================= */
(function method() {
  const STEPS = [
    { t: "A study, many views", k: "Input",
      x: "Each study combines videos from multiple views with a clinical report, which supplies a conclusion and findings for 12 cardiac structures. Videos are resampled, the sector extracted, padded to a square and resized to 224 × 224.",
      extra: `<div class="note">Pretraining: <b>60,782</b> studies, <b>2,651,674</b> videos and images (1,033,450 B-mode, 832,319 color Doppler, 785,905 stills) from 150 centers.</div>` },
    { t: "Space–time patch tokens", k: "Video encoder",
      x: "The video encoder is a TimeSformer with divided space–time attention, initialized from Kinetics-400. Each video gives N = 3,136 space–time patch tokens: 196 patches in each of 16 frames." },
    { t: "12 anatomical queries", k: "Anatomical queries",
      x: "12 learned anatomical queries attend to all patch tokens through an eight-head cross-attention layer, giving one embedding per cardiac structure. Nothing in the architecture ties query k to a structure: the video-structure loss trains query k to match the structure text of structure k.",
      extra: `<div class="note">Rings mark where each structure is on this CAMUS clip (expert masks for LV, LA, MV; placed by eye for RV, RA, TV). They show what each query is trained to capture, not learned attention maps.</div>` },
    { t: "Visibility-gated captions", k: "Visibility matrix",
      x: "A visibility matrix gives each view–structure pair a weight w ∈ {0, 0.1, 0.3, 0.5, 0.8, 1}. For each batch a threshold is drawn from 0 to 1; a caption contains the findings whose weight exceeds it. Each finding is then dropped with probability 0.2 and the rest are shuffled.",
      extra: `<button class="btn sm" id="openMatrix">View the full 66 × 12 matrix</button>` },
    { t: "Study transformer", k: "Study transformer",
      x: "A study transformer combines the videos of a study into one global and 12 anatomical study embeddings. Each anatomical query pools its structure's tokens only from videos whose view has positive visibility for that structure." },
    { t: "Four contrastive losses", k: "Losses",
      x: "Four CLIP losses match each kind of embedding with its text: each video with its caption, each anatomical embedding with its structure text, and each study with its report.",
      extra: `<div class="eqline" id="eqline">ℒ = <span class="t" data-k="video">ℒ<sub>video</sub></span> + <span class="t" data-k="vstruct">0.5 ℒ<sub>video-structure</sub></span> + <span class="t" data-k="study">0.5 ℒ<sub>study</sub></span> + <span class="t" data-k="sstruct">0.1 ℒ<sub>study-structure</sub></span></div><div class="note" id="lossNote"></div>` }
  ];
  let cur = 0, playing = false, timer = null, prog = null, cine = null;
  const stage = $("#mStage"), stepsEl = $("#mSteps");
  stepsEl.innerHTML = STEPS.map((s, i) => `<button data-i="${i}"><b>0${i + 1}</b>${s.t}<span class="bar"></span></button>`).join("");
  stepsEl.addEventListener("click", e => { const b = e.target.closest("button"); if (b) { stop(); go(+b.dataset.i); } });
  $("#mPrev").addEventListener("click", () => { stop(); go((cur + STEPS.length - 1) % STEPS.length); });
  $("#mNext").addEventListener("click", () => { stop(); go((cur + 1) % STEPS.length); });
  $("#mPlay").addEventListener("click", () => playing ? stop() : play());
  const DWELL = 9000;
  function play() { playing = true; $("#mPlay").textContent = "❚❚ Pause"; tickProg(); }
  function stop() { playing = false; $("#mPlay").textContent = "▶ Play"; clearTimeout(timer); cancelAnimationFrame(prog); stepsEl.querySelectorAll(".bar").forEach(b => b.style.width = 0); }
  function tickProg() {
    const t0 = performance.now(), bar = stepsEl.querySelectorAll(".bar")[cur];
    const f = now => { const k = Math.min(1, (now - t0) / DWELL); bar.style.width = (100 * k) + "%"; if (k < 1 && playing) prog = requestAnimationFrame(f); else if (playing) { bar.style.width = 0; go((cur + 1) % STEPS.length); tickProg(); } };
    prog = requestAnimationFrame(f);
  }
  let lossSel = "vstruct";
  function overview() {
    const el = $("#mOverview"); el.innerHTML = "";
    const W = 1200, H = 336;
    const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${W} ${H}`).attr("role", "img").attr("aria-label", "EchoAlign architecture");
    const defs = svg.append("defs");
    defs.append("marker").attr("id", "ovArr").attr("viewBox", "0 0 10 10").attr("refX", 9).attr("refY", 5).attr("markerWidth", 7).attr("markerHeight", 7).attr("orient", "auto-start-reverse")
      .append("path").attr("d", "M0,0 L10,5 L0,10 z").attr("fill", "var(--ink-3)");
    const hp = defs.append("pattern").attr("id", "ovHatch").attr("width", 4).attr("height", 4).attr("patternUnits", "userSpaceOnUse").attr("patternTransform", "rotate(45)");
    hp.append("rect").attr("width", 4).attr("height", 4).attr("fill", "var(--card)"); hp.append("line").attr("y2", 4).attr("stroke", "var(--ink-3)").attr("stroke-width", 1.1);
    const bands = svg.append("g"), hlLayer = svg.append("g"), arrows = svg.append("g"), body = svg.append("g");
    [[8, "Video level"], [176, "Study level"]].forEach(([y, t]) => {
      bands.append("rect").attr("x", 4).attr("y", y).attr("width", 860).attr("height", 156).attr("rx", 14).attr("fill", "var(--bg-2)").attr("stroke", "var(--rule)");
      bands.append("text").attr("x", 18).attr("y", y + 18).attr("fill", "var(--ink)").attr("font-size", 13).attr("font-weight", 600).text(t);
    });
    const arrow = (d, dash) => arrows.append("path").attr("d", d).attr("fill", "none").attr("stroke", "var(--ink-3)").attr("stroke-width", 1.3).attr("stroke-dasharray", dash ? "4 3" : null).attr("marker-end", "url(#ovArr)");
    const groups = [];
    const G = (step, extra = {}) => { const g = body.append("g").attr("class", "ov-g").style("cursor", "pointer"); groups.push({ g, step, ...extra }); return g; };
    const box = (g, x, y, w, h, t1, t2) => {
      g.append("rect").attr("x", x).attr("y", y).attr("width", w).attr("height", h).attr("rx", 10).attr("fill", "var(--card)").attr("stroke", "var(--ink-3)");
      const t = g.append("text").attr("x", x + w / 2).attr("y", y + h / 2 - 3).attr("text-anchor", "middle").attr("font-size", 13).attr("font-weight", 600).attr("fill", "var(--ink)").text(t1);
      const sub = g.append("text").attr("x", x + w / 2).attr("y", y + h / 2 + 14).attr("text-anchor", "middle").attr("font-size", 11.5).attr("fill", "var(--ink-2)").text(t2);
      // subtitle wider than the box: break it onto two lines and lift the title
      if (sub.node().getComputedTextLength() > w - 14 && t2.includes(" ")) {
        const words = t2.split(" "), cut = Math.ceil(words.length / 2);
        sub.text(null).attr("y", y + h / 2 + 8);
        sub.append("tspan").attr("x", x + w / 2).text(words.slice(0, cut).join(" "));
        sub.append("tspan").attr("x", x + w / 2).attr("dy", "1.2em").text(words.slice(cut).join(" "));
        t.attr("y", y + h / 2 - 9);
      }
    };
    const tokens = (g, x, y, sz = 12, gap = 3) => {
      g.append("rect").attr("x", x).attr("y", y + (sz + gap) / 2).attr("width", sz).attr("height", sz).attr("rx", 2).attr("fill", "var(--m0)");
      STRUCT.forEach((k, i) => g.append("rect").attr("x", x + (sz + gap) * (1 + i % 6)).attr("y", y + Math.floor(i / 6) * (sz + gap)).attr("width", sz).attr("height", sz).attr("rx", 2).attr("fill", SCX(k)));
    };
    const label = (g, x, y, t, sz = 11.5, col = "var(--ink-2)") => g.append("text").attr("x", x).attr("y", y).attr("text-anchor", "middle").attr("font-size", sz).attr("fill", col).text(t);
    const matrix = (g, x, y, col, mask, t) => {
      const n = 7, c = 11;
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        const m = mask && mask.includes(i);
        g.append("rect").attr("x", x + j * c).attr("y", y + i * c).attr("width", c - 1.5).attr("height", c - 1.5).attr("rx", 1.5)
          .attr("fill", m ? "url(#ovHatch)" : i === j ? col : "var(--card)").attr("stroke", i === j && !m ? col : "var(--rule)");
      }
      label(g, x + n * c / 2, y + n * c + 16, t, 12, "var(--ink)").attr("font-weight", 600);
    };
    const doc = (g, x, y, w, h, rows) => {
      g.append("path").attr("d", `M${x},${y} h${w - 12} l12,12 v${h - 12} h${-w} z`).attr("fill", "var(--card)").attr("stroke", "var(--ink-3)");
      const sp = Math.min(8, (h - 18) / rows.length);
      rows.forEach((c, i) => { g.append("rect").attr("x", x + 7).attr("y", y + 12 + i * sp).attr("width", 6).attr("height", Math.min(5, sp - 1.5)).attr("fill", c); g.append("rect").attr("x", x + 16).attr("y", y + 13 + i * sp).attr("width", w - 26).attr("height", 3).attr("rx", 1.5).attr("fill", "var(--rule)"); });
    };
    // 1 · video
    const g0 = G(0);
    [2, 1, 0].forEach(i => g0.append("image").attr("href", "assets/camus/a4c_frame.jpg").attr("x", 26 + i * 6).attr("y", 42 + i * 6).attr("width", 92).attr("height", 76).attr("preserveAspectRatio", "xMidYMid slice"));
    label(g0, 78, 145, "Echo videos");
    arrow("M132,82 L160,82");
    // 2 · TimeSformer
    const g1 = G(1); box(g1, 162, 54, 122, 56, "Video encoder", "TimeSformer"); label(g1, 223, 130, "3,136 patch tokens");
    arrow("M286,82 L312,82");
    // 3 · queries
    const g2 = G(2); tokens(g2, 316, 68); label(g2, 368, 116, "global + 12 anatomical"); label(g2, 368, 131, "12 cross-attention queries", 11.5, "var(--ink-3)");
    // 4 · captions / visibility / report (text side, video level)
    const g3 = G(3);
    doc(g3, 1010, 36, 52, 70, [SCX("LV"), SCX("MV"), SCX("TV"), SCX("LA")]);
    g3.append("rect").attr("x", 1074).attr("y", 30).attr("width", 26).attr("height", 104).attr("rx", 6).attr("fill", "none").attr("stroke", "var(--ink)").attr("stroke-dasharray", "3 3");
    STRUCT.forEach((k, i) => { const w = visOf("A4C")[k]; g3.append("rect").attr("x", 1080).attr("y", 36 + i * 8).attr("width", 14).attr("height", 6).attr("rx", 1).attr("fill", w > 0 ? SCX(k) : "var(--card)").attr("fill-opacity", w > 0 ? .3 + .7 * w : 1).attr("stroke", w > 0 ? "none" : "var(--rule)"); });
    doc(g3, 1112, 36, 60, 96, STRUCT.slice(0, 10).map(SCX));
    label(g3, 1036, 150, "Caption"); label(g3, 1087, 150, "Visibility"); label(g3, 1142, 150, "Report");
    // 5 · study level
    const g4 = G(4);
    for (let r = 0; r < 4; r++) tokens(g4, 22, 204 + r * 24, 7, 2);
    label(g4, 60, 318, "1–8 videos of a study");
    box(g4, 162, 228, 122, 56, "Study transformer", "visibility-masked pooling");
    tokens(g4, 316, 238); label(g4, 368, 290, "13 study embeddings");
    arrow("M368,138 C368,180 130,170 86,198", true);
    arrow("M86,256 L160,256"); arrow("M286,256 L312,256");
    // 6 · losses + text encoder
    const L = [["vstruct", 520, 30, SCX("LV"), [2, 5], "structure loss"], ["video", 680, 30, "var(--m1)", null, "video loss"],
               ["sstruct", 520, 198, SCX("AV"), [4], "study-structure loss"], ["study", 680, 198, "var(--m0)", null, "study loss"]];
    L.forEach(([k, x, y, col, mask, t]) => { const g = G(5, { loss: k }); matrix(g, x, y, col, mask, t); });
    arrow("M418,80 C460,80 470,68 512,68"); arrow("M334,66 C334,14 700,8 716,24");
    arrow("M418,248 C460,248 470,236 512,236"); arrow("M322,236 C322,180 700,176 716,192");
    const g5 = G(5);
    box(g5, 884, 104, 104, 130, "Text encoder", "PubMedBERT");
    const g5b = G(5); doc(g5b, 1050, 196, 66, 96, STRUCT.map(SCX)); label(g5b, 1083, 312, "Full report & structure findings");
    arrow("M1036,110 C1036,132 1012,138 992,138");
    arrow("M1048,244 C1020,244 1012,212 992,210");
    arrow("M882,140 C836,140 812,70 766,70"); arrow("M882,200 C836,200 812,238 766,238");
    // numbered badges + highlight rects
    const steps = new Map();
    groups.forEach(o => { if (!steps.has(o.step)) steps.set(o.step, []); steps.get(o.step).push(o); });
    groups.forEach(o => {
      const bb = o.g.node().getBBox();
      o.hl = hlLayer.append("rect").attr("x", bb.x - 7).attr("y", bb.y - 7).attr("width", bb.width + 14).attr("height", bb.height + 14).attr("rx", 10)
        .attr("fill", "var(--accent)").attr("fill-opacity", 0).attr("stroke", "var(--accent)").attr("stroke-opacity", 0).attr("stroke-width", 2);
      o.g.insert("rect", ":first-child").attr("x", bb.x - 7).attr("y", bb.y - 7).attr("width", bb.width + 14).attr("height", bb.height + 14).attr("fill", "transparent");
      o.g.on("click", () => { stop(); if (o.loss) lossSel = o.loss; go(o.step); })
        .on("mouseenter", () => o.hl.attr("stroke-opacity", .55).attr("stroke-dasharray", "4 3"))
        .on("mouseleave", () => paint())
        .on("mousemove", ev => showTip(`<div class="tt">0${o.step + 1} · ${STEPS[o.step].t}</div>${o.loss ? { video: "Video loss", vstruct: "Video-structure loss", study: "Study loss", sstruct: "Study-structure loss" }[o.loss] + " · " : ""}click to see it in detail`, ev))
        .on("mouseout", hideTip);
    });
    steps.forEach((arr, st) => {
      const bb = arr[0].g.node().getBBox();
      const bx = st === 5 ? 512 : st === 4 ? 300 : bb.x + bb.width + 2, by = st === 5 ? 22 : st === 4 ? 230 : bb.y - 2;
      const b = body.append("g").attr("pointer-events", "none");
      b.append("circle").attr("cx", bx).attr("cy", by).attr("r", 10).attr("fill", "var(--ink)");
      b.append("text").attr("x", bx).attr("y", by + 4).attr("text-anchor", "middle").attr("font-size", 11).attr("font-weight", 700).attr("fill", "var(--bg)").text(st + 1);
    });
    function paint() {
      groups.forEach(o => {
        const on = o.step === cur && (!o.loss || cur !== 5 || o.loss === lossSel);
        const inStep = o.step === cur;
        o.hl.attr("stroke-dasharray", null).attr("stroke-opacity", on ? .95 : 0).attr("fill-opacity", inStep ? .07 : 0);
        o.g.style("opacity", inStep ? 1 : .55);
      });
    }
    overview.paint = paint; paint();
  }
  function go(i) {
    cur = i;
    stepsEl.querySelectorAll("button").forEach((b, j) => b.classList.toggle("on", j === i));
    const S = STEPS[i];
    $("#mNum").textContent = `STEP 0${i + 1} / 0${STEPS.length}`; $("#mTitle").textContent = S.t; $("#mText").textContent = S.x; $("#mSideExtra").innerHTML = S.extra || "";
    overview.paint ? overview.paint() : overview();
    stage.innerHTML = ""; cine = null;
    [s1, s2, s3, s4, s5, s6][i]();
  }
  const layer = (html = "") => { const d = document.createElement("div"); d.className = "layer"; d.innerHTML = html; stage.appendChild(d); return d; };
  const centerBox = (d, maxW) => { d.style.display = "grid"; d.style.placeItems = "center"; d.style.padding = "14px"; const inner = document.createElement("div"); inner.style.width = "100%"; inner.style.maxWidth = maxW + "px"; d.appendChild(inner); return inner; };

  // 1. study: real clips + schematic slices of other views
  function s1() {
    const d = layer(); d.style.padding = "16px"; d.style.display = "grid"; d.style.gap = "10px"; d.style.alignContent = "center"; d.style.justifyContent = "center";
    { // size tiles so two rows fit the stage height
      const cols = stage.clientWidth < 640 ? 2 : 3, rows = 6 / cols, W0 = stage.clientWidth - 32 - 10 * (cols - 1), H0 = stage.clientHeight - 32 - 10 * (rows - 1);
      const tw = stage.clientWidth < 900 ? W0 / cols : Math.min(W0 / cols, H0 / rows * 1.21);
      d.style.gridTemplateColumns = `repeat(${cols}, ${Math.floor(tw)}px)`; }
    const items = [["A4C", "cine"], ["A2C", "cine"], ["PLAX Standard", "slice"], ["PSAX Papillary", "slice"], ["Subcostal Standard", "slice"], ["Suprasternal Notch", "slice"]];
    items.forEach(([v, kind], j) => {
      const c = document.createElement("div");
      c.style.cssText = `position:relative;border-radius:10px;overflow:hidden;background:#000;aspect-ratio:1.21/1;opacity:0;transform:translateY(10px);transition:all .5s ${j * 0.12}s`;
      c.innerHTML = `<div style="position:absolute;left:8px;top:6px;z-index:2;font:600 11px var(--mono);color:#e6e9ed">${v}<br><span style="color:#9aa3ae;font-weight:400">${kind === "slice" ? "3D-model slice" : "CAMUS clip"}</span></div>`;
      d.appendChild(c);
      if (kind === "cine") { const h = document.createElement("div"); c.appendChild(h); makeCine(h, { view: v, mode: "plain", controls: false }); h.querySelector(".cine").style.height = "100%"; h.querySelector("video").style.cssText = "width:100%;height:100%;object-fit:cover"; h.style.height = "100%"; }
      else { const h = document.createElement("div"); h.style.cssText = "width:100%;height:100%"; c.appendChild(h); drawSlice(h, v); }
      requestAnimationFrame(() => requestAnimationFrame(() => { c.style.opacity = 1; c.style.transform = "none"; }));
    });
    const rep = layer(`<div style="position:absolute;right:14px;bottom:12px;background:var(--st-panel);border:1px solid var(--st-line);box-shadow:var(--shadow);border-radius:10px;padding:8px 10px;font:11px/1.5 var(--mono);color:var(--st-ink2);max-width:260px;opacity:0;transition:opacity .6s 1s" class="rp">Report · 12 structure findings + conclusion<br>${STRUCT.map(s => `<span style="color:${SCX(s)}">■</span>`).join(" ")}</div>`);
    requestAnimationFrame(() => requestAnimationFrame(() => rep.querySelector(".rp").style.opacity = 1));
  }
  // 2. patch tokens
  function s2() {
    const d = layer(); d.style.display = "grid"; d.style.gridTemplateColumns = stage.clientWidth < 640 ? "1fr" : "1fr 230px"; d.style.gap = "16px"; d.style.padding = "14px"; d.style.alignItems = "center";
    const left = document.createElement("div"); left.style.cssText = `width:100%;max-width:${Math.max(220, stage.clientHeight - 28)}px;justify-self:center`; d.appendChild(left);
    cine = makeCine(left, { view: "A4C", mode: "patches", controls: false });
    const right = document.createElement("div"); d.appendChild(right);
    const svg = d3.select(right).append("svg").attr("viewBox", "0 0 230 330").style("position", "static");
    for (let f = 15; f >= 0; f--) {
      svg.append("rect").attr("x", 20 + f * 6).attr("y", 30 + f * 9).attr("width", 110).attr("height", 90).attr("rx", 4).attr("fill", "var(--st-panel2)").attr("stroke", "var(--st-blue)").attr("stroke-opacity", .55)
        .attr("opacity", 0).transition().delay(reduceMotion ? 0 : 200 + (15 - f) * 70).duration(300).attr("opacity", 1);
    }
    svg.append("text").attr("x", 115).attr("y", 270).attr("text-anchor", "middle").attr("fill", "var(--st-ink)").attr("font-size", 15).attr("font-family", "var(--mono)").text("16 frames × 196 patches");
    const n = svg.append("text").attr("x", 115).attr("y", 300).attr("text-anchor", "middle").attr("fill", "var(--st-blue)").attr("font-size", 24).attr("font-weight", 600).attr("font-family", "var(--mono)").text("0");
    n.transition().delay(300).duration(reduceMotion ? 0 : 1600).tween("t", () => t => n.text(d3.format(",")(Math.round(3136 * t)) + " tokens"));
  }
  // 3. anatomical queries on the CAMUS clip
  function s3() {
    const d = layer(); d.style.padding = "12px"; const inner = document.createElement("div"); inner.style.maxWidth = Math.min(900, (stage.clientHeight - 80) * 1.212) + "px"; inner.style.margin = "0 auto"; d.appendChild(inner);
    cine = makeCine(inner, { view: "A4C", mode: "leaders", queriesLeft: false });
  }
  // 4. visibility + caption sampler
  function s4() {
    const d = layer(); d.style.display = "grid"; d.style.gridTemplateColumns = stage.clientWidth < 640 ? "1fr" : "1fr 1fr"; d.style.gap = "16px"; d.style.padding = "14px"; d.style.alignItems = "start"; if (stage.clientWidth < 640) { d.style.position = "relative"; }
    const left = document.createElement("div"); d.appendChild(left);
    cine = makeCine(left, { view: "A4C", mode: "rings", onView: v => sampler.setView(v) });
    const right = document.createElement("div"); right.style.color = "var(--st-ink2)"; d.appendChild(right);
    const sampler = captionSampler(right, "A4C");
    $("#openMatrix")?.addEventListener("click", openMatrix);
  }
  // 5. study transformer with visibility-masked pooling
  function s5() {
    const d = layer(); d.style.padding = "10px";
    const vids = ["A4C", "A2C", "PLAX Standard", "PSAX Papillary", "Suprasternal Notch"];
    const short = { "A4C": "A4C", "A2C": "A2C", "PLAX Standard": "PLAX", "PSAX Papillary": "PSAX", "Suprasternal Notch": "SSN" };
    let sel = "AoR";
    const chips = document.createElement("div"); chips.className = "chips"; chips.style.cssText = "justify-content:center;margin:2px 0 6px";
    chips.innerHTML = STRUCT.map(s => `<button class="chip" data-s="${s}" style="background:var(--st-panel);border-color:var(--st-line);color:var(--st-ink2)"><span style="color:${SCX(s)}">■</span> ${s}</button>`).join("");
    d.appendChild(chips);
    const host = document.createElement("div"); d.appendChild(host);
    chips.addEventListener("click", e => { const b = e.target.closest(".chip"); if (b) { sel = b.dataset.s; draw(); } });
    function draw() {
      chips.querySelectorAll(".chip").forEach(c => { c.style.borderColor = c.dataset.s === sel ? SCX(sel) : "var(--st-line)"; c.style.color = c.dataset.s === sel ? "var(--st-ink)" : "var(--st-ink2)"; });
      host.innerHTML = "";
      const W = 900, H = 400, svg = d3.select(host).append("svg").attr("viewBox", `0 0 ${W} ${H}`).style("position", "static").style("width", "100%");
      const colX = i => 70 + i * 120, tok = 12;
      const outY = 330, outX0 = 300;
      // output study embeddings
      const outs = ["G", ...STRUCT];
      outs.forEach((s, j) => svg.append("rect").attr("x", outX0 + j * (tok + 9)).attr("y", outY).attr("width", tok + 4).attr("height", tok + 4).attr("rx", 3)
        .attr("fill", s === "G" ? "var(--st-blue)" : SCX(s)).attr("opacity", s === sel ? 1 : .35).attr("stroke", s === sel ? "var(--st-ink)" : "none").attr("stroke-width", 1.5));
      svg.append("text").attr("x", outX0 + 7 * (tok + 9)).attr("y", outY + 40).attr("text-anchor", "middle").attr("fill", "var(--st-ink2)").attr("font-size", 13).attr("font-family", "var(--sans)").text("13 study embeddings: 1 global + 12 anatomical");
      const selX = outX0 + (outs.indexOf(sel)) * (tok + 9) + (tok + 4) / 2;
      // transformer box
      svg.append("rect").attr("class", "stbox").attr("x", 120).attr("y", 200).attr("width", 660).attr("height", 96).attr("rx", 12).attr("fill", "var(--st-panel)").attr("stroke", "var(--st-line)");
      svg.append("text").attr("class", "stbox").attr("x", 450).attr("y", 230).attr("text-anchor", "middle").attr("fill", "var(--st-ink)").attr("font-size", 15).attr("font-weight", 600).attr("font-family", "var(--sans)").text("Study transformer");
      svg.append("text").attr("class", "stbox").attr("x", 450).attr("y", 250).attr("text-anchor", "middle").attr("fill", "var(--st-ink3)").attr("font-size", 12).attr("font-family", "var(--sans)").text("2 self-attention layers · 13 pooling queries · videos as an unordered set");
      vids.forEach((v, i) => {
        const x = 130 + i * 150, w = visOf(v)[sel];
        svg.append("text").attr("x", x + 50).attr("y", 26).attr("text-anchor", "middle").attr("fill", "var(--st-ink)").attr("font-size", 13).attr("font-weight", 600).attr("font-family", "var(--mono)").text(short[v]);
        svg.append("text").attr("x", x + 50).attr("y", 44).attr("text-anchor", "middle").attr("fill", w > 0 ? SCX(sel) : "var(--st-ink3)").attr("font-size", 12).attr("font-family", "var(--mono)").text(`w(${sel}) = ${w}`);
        // 13 tokens per video
        ["G", ...STRUCT].forEach((s, j) => {
          const tx = x + (j % 7) * 15, ty = 60 + Math.floor(j / 7) * 15;
          svg.append("rect").attr("x", tx).attr("y", ty).attr("width", 12).attr("height", 12).attr("rx", 2).attr("fill", s === "G" ? "var(--st-blue)" : SCX(s)).attr("opacity", s === sel ? 1 : .3);
        });
        const sx = x + (STRUCT.indexOf(sel) + 1) % 7 * 15 + 6, sy = 60 + Math.floor((STRUCT.indexOf(sel) + 1) / 7) * 15 + 12;
        const p = svg.append("path").attr("d", `M${sx},${sy} C${sx},${sy + 120} ${selX},${outY - 110} ${selX},${outY - 2}`).attr("fill", "none")
          .attr("stroke", w > 0 ? SCX(sel) : "var(--st-mute)").attr("stroke-width", w > 0 ? 2.2 : 1.2).attr("stroke-dasharray", w > 0 ? "6 5" : "2 5").attr("opacity", w > 0 ? .95 : .5).attr("class", w > 0 ? "flow" : "");
        if (w === 0) svg.append("text").attr("class", "mlab").attr("x", x + 50).attr("y", 112).attr("text-anchor", "middle").attr("fill", "var(--st-ink3)").attr("font-size", 11).attr("font-family", "var(--sans)").attr("paint-order", "stroke").attr("stroke", "var(--st-bg)").attr("stroke-width", 5).attr("stroke-linejoin", "round").text("masked out of this query");
      });
      const nUsed = vids.filter(v => visOf(v)[sel] > 0).length;
      svg.append("text").attr("class", "stbox").attr("x", 450).attr("y", 278).attr("text-anchor", "middle").attr("fill", SCX(sel)).attr("font-size", 13).attr("font-weight", 600).attr("font-family", "var(--sans)").text(`${SNAME[sel]} study embedding pools ${nUsed} of ${vids.length} videos`);
      svg.selectAll(".stbox, .mlab").raise();   // flows pass behind the transformer box, never over its text
      if (!reduceMotion) { let off = 0; const t = d3.timer(() => { if (!host.isConnected) { t.stop(); return; } off -= .6; svg.selectAll(".flow").attr("stroke-dashoffset", off); }); }
    }
    draw();
  }
  // 6. losses
  function s6() {
    const d = layer(); d.style.padding = "14px";
    const L = {
      video: { name: "Video loss", w: "× 1", col: "#5f88b8", left: "Global video embedding", right: "Video caption", ex: "Color Doppler video. A4C view. Estimated EF 55-60%. Leaflets open well. Mild mitral regurgitation.", note: "One matrix per batch: each global video embedding is pulled toward its own sampled caption and pushed away from the other captions. Pairs with identical captions are removed from the negatives; videos from the same study otherwise remain negatives." },
      vstruct: { name: "Video-structure loss", w: "× 0.5", col: "#B8322A", left: "Anatomical video embedding (structure k)", right: "Structure text of k", ex: "Mitral valve: Leaflets open well. Mild mitral regurgitation.", note: "Twelve matrices, one per structure. A video's anatomical embedding of structure k enters matrix k only if its view has w > 0 and its report has a finding for k; otherwise its row and column are masked.", mask: [2, 5] },
      study: { name: "Study loss", w: "× 0.5", col: "var(--st-blue)", left: "Global study embedding", right: "Study text", ex: "Conclusion + all non-empty findings in a fixed order.", note: "Each global study embedding (pooled by the study transformer over all its videos) is matched with its study text: the conclusion and all non-empty findings in a fixed order. Studies with identical text form one positive set with equal target probability." },
      sstruct: { name: "Study-structure loss", w: "× 0.1", col: "#B7791F", left: "Anatomical study embedding (structure k)", right: "Structure text of k", ex: "Aortic valve: Three cusps with mild sclerosis; no stenosis.", note: "A structure is included when at least one video of the study has positive visibility for it; a missing finding is matched with the encoding of an empty finding.", mask: [4] }
    };
    let sel = lossSel, stackTimer = null;
    const host = document.createElement("div"); d.appendChild(host);
    function draw() {
      if (stackTimer) { stackTimer.stop(); stackTimer = null; }
      lossSel = sel; overview.paint && overview.paint();
      document.querySelectorAll("#eqline .t").forEach(t => t.classList.toggle("on", t.dataset.k === sel));
      $("#lossNote") && ($("#lossNote").textContent = L[sel].note);
      host.innerHTML = "";
      const W = 900, H = 470, svg = d3.select(host).append("svg").attr("viewBox", `0 0 ${W} ${H}`).style("position", "static").style("width", "100%");
      const defs = svg.append("defs"); const pat = defs.append("pattern").attr("id", "hatchM").attr("width", 5).attr("height", 5).attr("patternUnits", "userSpaceOnUse").attr("patternTransform", "rotate(45)");
      pat.append("rect").attr("width", 5).attr("height", 5).attr("fill", "var(--st-panel)"); pat.append("line").attr("y2", 5).attr("stroke", "var(--st-ink3)").attr("stroke-width", 1.4);
      // tabs as small buttons in svg
      Object.entries(L).forEach(([k, v], i) => {
        const g = svg.append("g").attr("transform", `translate(${20 + i * 218},10)`).style("cursor", "pointer").on("click", () => { sel = k; draw(); });
        g.append("rect").attr("width", 208).attr("height", 34).attr("rx", 9).attr("fill", k === sel ? "var(--st-panel2)" : "var(--st-panel)").attr("stroke", k === sel ? v.col : "var(--st-line)").attr("stroke-width", k === sel ? 2 : 1);
        g.append("rect").attr("x", 10).attr("y", 12).attr("width", 10).attr("height", 10).attr("rx", 2).attr("fill", v.col);
        g.append("text").attr("x", 28).attr("y", 22).attr("fill", "var(--st-ink)").attr("font-size", 13).attr("font-family", "var(--sans)").text(v.name);
        g.append("text").attr("x", 198).attr("y", 22).attr("text-anchor", "end").attr("fill", "var(--st-ink3)").attr("font-size", 11).attr("font-family", "var(--mono)").text(v.w);
      });
      if (sel === "vstruct" || sel === "sstruct") { drawStack(svg, sel); return; }
      if (sel === "video") { drawVideoLoss(svg); return; }
      if (sel === "study") { drawStudyLoss(svg); return; }
      const v = L[sel], n = 8, c = 30, x0 = 330, y0 = 135;
      for (let i = 0; i < n; i++) {
        const masked = v.mask && v.mask.includes(i);
        svg.append("rect").attr("x", x0 - 40).attr("y", y0 + i * c + 6).attr("width", 20).attr("height", 18).attr("rx", 3).attr("fill", masked ? "var(--st-line)" : v.col).attr("opacity", masked ? .6 : .9);
        svg.append("rect").attr("x", x0 + i * c + 6).attr("y", y0 - 40).attr("width", 18).attr("height", 20).attr("rx", 3).attr("fill", "#d9d4c7").attr("opacity", .85);
        for (let j = 0; j < n; j++) {
          const diag = i === j && !masked;
          svg.append("rect").attr("x", x0 + j * c).attr("y", y0 + i * c).attr("width", c - 3).attr("height", c - 3).attr("rx", 3)
            .attr("fill", masked ? "url(#hatchM)" : diag ? v.col : "var(--st-panel)").attr("stroke", "var(--st-line)")
            .attr("opacity", diag ? 0 : 1).transition().delay(reduceMotion ? 0 : 300 + i * 120).duration(350).attr("opacity", 1);
        }
      }
      svg.append("text").attr("x", x0 - 50).attr("y", y0 + n * c / 2).attr("text-anchor", "end").attr("fill", "var(--st-ink2)").attr("font-size", 13).attr("font-family", "var(--sans)").text(v.left);
      svg.append("text").attr("x", x0 + n * c / 2).attr("y", y0 - 52).attr("text-anchor", "middle").attr("fill", "var(--st-ink2)").attr("font-size", 13).attr("font-family", "var(--sans)").text(v.right + " (PubMedBERT)");
      svg.append("text").attr("x", x0 + n * c + 20).attr("y", y0 + 20).attr("fill", "var(--st-ink)").attr("font-size", 13).attr("font-family", "var(--sans)").attr("font-weight", 600).text("pull matched pairs together (diagonal)");
      svg.append("text").attr("x", x0 + n * c + 20).attr("y", y0 + 40).attr("fill", "var(--st-ink3)").attr("font-size", 12.5).attr("font-family", "var(--sans)").text("push mismatched pairs apart");
      svg.append("text").attr("x", x0 + n * c + 20).attr("y", y0 + 70).attr("fill", "var(--st-ink3)").attr("font-size", 12).attr("font-family", "var(--sans)").text("shared learned logit scale ∈ [1, 32]");
      const fo = svg.append("foreignObject").attr("x", x0 + n * c + 16).attr("y", y0 + 100).attr("width", W - (x0 + n * c + 30)).attr("height", 150);
      fo.append("xhtml:div").attr("style", "font:12px/1.5 var(--mono);color:var(--st-ink2);background:var(--st-panel);border:1px solid var(--st-line);border-radius:8px;padding:8px 10px").text("e.g. “" + v.ex + "”");
      svg.append("text").attr("x", W / 2).attr("y", H - 14).attr("text-anchor", "middle").attr("fill", "var(--st-ink3)").attr("font-size", 12).attr("font-family", "var(--sans)").text("512-d embeddings, ℓ2-normalized; a batch of 1,024 videos across 64 GPUs");
    }


    // ----- shared helpers for the loss views -----
    function playToggle(svg, x, y, get, set) {
      const g = svg.append("g").attr("transform", `translate(${x},${y})`).style("cursor", "pointer");
      const r = g.append("rect").attr("width", 84).attr("height", 24).attr("rx", 12).attr("fill", "var(--st-ink)");
      const t = g.append("text").attr("x", 42).attr("y", 16).attr("text-anchor", "middle").attr("font-size", 12).attr("font-weight", 600).attr("font-family", "var(--sans)").attr("fill", "var(--st-bg)");
      const sync = () => t.text(get() ? "❚❚  Pause" : "▶  Play");
      g.on("click", () => { set(!get()); sync(); });
      sync();
      return { sync };
    }
    const rng = seed => () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
    const NORMAL = { LV: "Normal size and systolic function.", LA: "Normal size.", RV: "Normal size and function.", RA: "Normal size.", AV: "Trileaflet; no stenosis or regurgitation.", MV: "Normal appearance.", TV: "Normal appearance.", PV: "Normal appearance.", AoR: "Normal dimensions.", Arch: "", PA: "Normal pressure.", Peri: "No pericardial effusion." };
    const REPORTS = {
      A: { conc: CONCLUSION, f: REPORT },
      B: { conc: "Moderate mitral regurgitation. Mildly dilated left atrium.", f: { ...REPORT, MV: "Thickened leaflets. Moderate mitral regurgitation.", LA: "Mildly dilated." } },
      C: { conc: "Normal study.", f: NORMAL },
      D: { conc: "Moderate tricuspid regurgitation. Elevated pulmonary pressure.", f: { ...NORMAL, TV: "Moderate tricuspid regurgitation.", PA: "Systolic pressure 48 mmHg.", RA: "Mildly dilated." } },
      E: { conc: "Mild aortic stenosis. Preserved LV function.", f: { ...NORMAL, AV: "Calcified cusps with mild stenosis.", AoR: "Mildly dilated aortic root." } }
    };
    const shortV = v => ({ "PLAX Standard": "PLAX", "PSAX Papillary": "PSAX papillary", "SUB IVC": "Subcostal IVC", "Suprasternal Notch": "Suprasternal" }[v] || v);
    function caption(view, rep, t, rand) {
      const w = visOf(view);
      let inc = STRUCT.filter(k => rep.f[k] && w[k] > t);
      if (!inc.length) return { text: rep.conc, inc: [], conc: true };
      if (inc.length > 1) { let kept = inc.filter(() => rand() >= .2); if (!kept.length) kept = [inc[0]]; inc = kept; }
      inc = inc.map(k => [k, rand()]).sort((a, b) => a[1] - b[1]).map(d => d[0]);
      const mod = view === "Pedoff" ? "Spectral Doppler image." : /Color/.test(view) ? "Color Doppler video." : "Grayscale B-mode video.";
      return { text: `${mod} ${shortV(view)} view. ` + inc.map(k => rep.f[k]).join(" "), inc, conc: false };
    }
    function targetBars(g, x, y, w, probs, col, label) {
      g.append("text").attr("x", x).attr("y", y - 6).attr("fill", "var(--st-ink2)").attr("font-size", 11.5).attr("font-family", "var(--sans)").text(label);
      const cw = w / probs.length;
      probs.forEach((p, j) => {
        g.append("rect").attr("x", x + j * cw + 2).attr("y", y).attr("width", cw - 4).attr("height", 30).attr("rx", 3).attr("fill", "var(--st-panel)");
        if (p === null) g.append("rect").attr("x", x + j * cw + 2).attr("y", y).attr("width", cw - 4).attr("height", 30).attr("rx", 3).attr("fill", "url(#hatchM)");
        else if (p > 0) g.append("rect").attr("x", x + j * cw + 2).attr("y", y + 30 * (1 - p)).attr("width", cw - 4).attr("height", 30 * p).attr("rx", 3).attr("fill", col).attr("opacity", 0)
          .transition().duration(reduceMotion ? 0 : 400).attr("opacity", 1);
        g.append("text").attr("x", x + j * cw + cw / 2).attr("y", y + 44).attr("text-anchor", "middle").attr("fill", "var(--st-ink3)").attr("font-size", 10.5).attr("font-family", "var(--mono)")
          .text(p === null ? "—" : p === 0 ? "0" : p === 1 ? "1" : "½");
      });
    }

    // ----- video loss: global video embeddings vs sampled captions -----
    function drawVideoLoss(svg) {
      const batch = [{ v: "A4C", st: "A" }, { v: "A2C", st: "A" }, { v: "PLAX Standard", st: "B" }, { v: "PSAX Papillary", st: "B" }, { v: "Suprasternal Notch", st: "C" }, { v: "SUB IVC", st: "D" }];
      const n = batch.length, cs = 38, x0 = 300, y0 = 172, col = "#5f88b8";
      let seed = 7, t = .42, sel = 0, auto = true;
      svg.append("text").attr("x", 20).attr("y", 72).attr("fill", "var(--st-ink3)").attr("font-size", 11.5).attr("font-family", "var(--sans)")
        .text("One matrix per batch · rows: global video embeddings · columns: each video's sampled caption");
      const tog = playToggle(svg, 800, 58, () => auto, v => { auto = v; });
      const rs = svg.append("g").attr("transform", "translate(700,58)").style("cursor", "pointer").on("click", () => { seed++; build(); });
      rs.append("rect").attr("width", 92).attr("height", 24).attr("rx", 12).attr("fill", "var(--st-panel)").attr("stroke", "var(--st-line)");
      rs.append("text").attr("x", 46).attr("y", 16).attr("text-anchor", "middle").attr("fill", "var(--st-ink)").attr("font-size", 12).attr("font-family", "var(--sans)").text("⟳ New batch");
      const root = svg.append("g");
      let caps = [];
      function build() {
        const rand = rng(seed); t = Math.round((0.15 + 0.7 * rand()) * 100) / 100;
        caps = batch.map(b => caption(b.v, REPORTS[b.st], t, rand));
        render();
      }
      function render() {
        root.selectAll("*").remove();
        root.append("text").attr("x", 20).attr("y", 104).attr("fill", "var(--st-ink)").attr("font-size", 12.5).attr("font-family", "var(--sans)")
          .text(`Batch threshold t = ${t.toFixed(2)}: a finding enters a caption when its visibility weight exceeds t; each is then dropped with p = 0.2 and the rest shuffled.`);
        const same = (i, j) => caps[i].text === caps[j].text;
        // column caption icons
        caps.forEach((c, j) => {
          const g = root.append("g").attr("transform", `translate(${x0 + j * cs + cs / 2 - 9},${y0 - 32})`);
          g.append("path").attr("d", "M0,0 h13 l5,5 v19 h-18 z").attr("fill", "#d9d4c7");
          (c.conc ? ["var(--st-ink3)"] : c.inc).slice(0, 4).forEach((k, q) => g.append("rect").attr("x", 3).attr("y", 6 + q * 4).attr("width", 11).attr("height", 2.4).attr("fill", c.conc ? k : SCX(k)));
        });
        root.append("text").attr("x", x0 + n * cs / 2).attr("y", y0 - 42).attr("text-anchor", "middle").attr("fill", "var(--st-ink2)").attr("font-size", 12).attr("font-family", "var(--sans)").text("sampled caption of each video");
        batch.forEach((b, i) => {
          const y = y0 + i * cs, g = root.append("g").style("cursor", "pointer").on("click", () => { sel = i; auto = false; tog.sync(); render(); });
          g.append("rect").attr("x", 14).attr("y", y + 1).attr("width", x0 - 22).attr("height", cs - 2).attr("rx", 6).attr("fill", i === sel ? "var(--st-panel2)" : "transparent");
          g.append("text").attr("x", 26).attr("y", y + cs / 2 - 2).attr("fill", "var(--st-ink)").attr("font-size", 13).attr("font-weight", 600).attr("font-family", "var(--sans)").text(shortV(b.v) + " view");
          g.append("text").attr("x", 26).attr("y", y + cs / 2 + 12).attr("fill", "var(--st-ink3)").attr("font-size", 10.5).attr("font-family", "var(--mono)").text(`study ${b.st}${caps[i].conc ? " · conclusion" : ""}`);
          g.append("rect").attr("x", x0 - 24).attr("y", y + cs / 2 - 7).attr("width", 14).attr("height", 14).attr("rx", 3).attr("fill", col);
          for (let j = 0; j < n; j++) {
            const diag = i === j, ident = !diag && same(i, j), sameStudy = !diag && !ident && batch[i].st === batch[j].st;
            root.append("rect").attr("x", x0 + j * cs + 1.5).attr("y", y + 1.5).attr("width", cs - 3).attr("height", cs - 3).attr("rx", 3)
              .attr("fill", diag ? col : ident ? "url(#hatchM)" : "var(--st-panel)").attr("stroke", sameStudy ? "#d9a441" : "var(--st-line)").attr("stroke-width", sameStudy ? 1.6 : 1)
              .attr("opacity", i === sel || j === sel ? 1 : .55);
          }
        });
        root.append("rect").attr("x", x0 - 2).attr("y", y0 + sel * cs - 2).attr("width", n * cs + 4).attr("height", cs + 4).attr("rx", 5).attr("fill", "none").attr("stroke", "var(--st-ink)").attr("stroke-width", 1.5);
        // legend
        const lg = root.append("g").attr("transform", `translate(${x0},${y0 + n * cs + 8})`);
        const anyIdent = caps.some((_, i) => caps.some((_, j) => i !== j && same(i, j)));
        [["#5f88b8", null, "positive pair"], ["var(--st-panel)", "#d9a441", "same study: still a negative"], ...(anyIdent ? [["url(#hatchM)", null, "identical caption: removed"]] : [])].forEach(([f, st, tx], q) => {
          lg.append("rect").attr("y", q * 15).attr("width", 11).attr("height", 11).attr("rx", 2).attr("fill", f).attr("stroke", st || "var(--st-line)").attr("stroke-width", st ? 1.6 : 1);
          lg.append("text").attr("x", 17).attr("y", q * 15 + 9).attr("fill", "var(--st-ink3)").attr("font-size", 11).attr("font-family", "var(--sans)").text(tx);
        });
        // right panel: the selected row
        const rx = x0 + n * cs + 40, rw = 900 - rx - 10, c = caps[sel];
        const fo = root.append("foreignObject").attr("x", rx).attr("y", y0 - 46).attr("width", rw).attr("height", 200);
        const div = fo.append("xhtml:div").attr("style", "font:12px/1.5 var(--sans);color:var(--st-ink2)");
        div.append("xhtml:div").attr("style", "color:var(--st-ink);font-weight:600;margin-bottom:4px").text(`Row ${sel + 1}: ${shortV(batch[sel].v)}, study ${batch[sel].st}`);
        const cap = div.append("xhtml:div").attr("style", "font:11px/1.5 var(--mono);background:var(--st-panel);border:1px solid var(--st-line);border-radius:8px;padding:7px 9px;max-height:150px;overflow:hidden");
        if (c.conc) cap.html(`<span style="color:var(--st-ink3)">no finding passes t → report conclusion:</span><br>${c.text}`);
        else { const parts = c.text.split(" view. "); cap.html(`<span style="color:var(--st-ink3)">${parts[0]} view.</span> ` + c.inc.map(k => `<span style="background:${SCX(k)}33;border-radius:3px;padding:0 2px">${REPORTS[batch[sel].st].f[k]}</span>`).join(" ")); }
        const probs = caps.map((_, j) => j === sel ? 1 : same(sel, j) ? null : 0);
        targetBars(root.append("g"), rx, y0 + 180, rw, probs, col, "Training target for this row (softmax over the captions)");
        const ident = caps.map((_, j) => j !== sel && same(sel, j)).some(Boolean);
        root.append("foreignObject").attr("x", rx).attr("y", y0 + 236).attr("width", rw).attr("height", 70).append("xhtml:div")
          .attr("style", "font:11px/1.45 var(--sans);color:var(--st-ink3)")
          .text(ident ? "Another video has the identical caption, so that pair is removed from the negatives rather than pushed apart."
                      : batch.some((b, j) => j !== sel && b.st === batch[sel].st) ? "The other video of the same study has a different caption, so it remains a negative." : "All other captions are negatives.");
      }
      build();
      if (!reduceMotion) stackTimer = d3.interval(() => { if (!svg.node().isConnected) { stackTimer && stackTimer.stop(); return; } if (auto) { sel = (sel + 1) % n; render(); } }, 2600);
    }

    // ----- study loss: global study embeddings vs study texts -----
    function drawStudyLoss(svg) {
      const studies = [{ name: "Study 1", views: ["A4C", "A2C", "PLAX Standard"], r: "A" }, { name: "Study 2", views: ["PLAX Standard", "PSAX Papillary"], r: "C" },
                       { name: "Study 3", views: ["A4C", "A5C"], r: "B" }, { name: "Study 4", views: ["A4C", "SUB IVC"], r: "E" }, { name: "Study 5", views: ["A4C", "PSAX Papillary"], r: "D" }];
      const n = studies.length, cs = 40, x0 = 300, y0 = 172, col = "var(--st-blue)";
      let sel = 0, auto = true;
      const text = s => { const R = REPORTS[s.r]; return R.conc + " " + STRUCT.filter(k => R.f[k]).map(k => R.f[k]).join(" "); };
      const texts = studies.map(text), same = (i, j) => texts[i] === texts[j];
      svg.append("text").attr("x", 20).attr("y", 72).attr("fill", "var(--st-ink3)").attr("font-size", 11.5).attr("font-family", "var(--sans)")
        .text("One matrix per batch · rows: global study embeddings · columns: each study's text (conclusion + all findings)");
      const tog = playToggle(svg, 800, 58, () => auto, v => { auto = v; });
      const root = svg.append("g");
      function render() {
        root.selectAll("*").remove();
        root.append("text").attr("x", 20).attr("y", 104).attr("fill", "var(--st-ink)").attr("font-size", 12.5).attr("font-family", "var(--sans)")
          .text("The study transformer's global query pools all videos of a study into one embedding; it is matched with the whole report.");
        texts.forEach((tx, j) => {
          const g = root.append("g").attr("transform", `translate(${x0 + j * cs + cs / 2 - 9},${y0 - 34})`);
          g.append("path").attr("d", "M0,0 h13 l5,5 v21 h-18 z").attr("fill", "#d9d4c7");
          for (let q = 0; q < 5; q++) g.append("rect").attr("x", 3).attr("y", 6 + q * 3.6).attr("width", q ? 11 : 12).attr("height", 2).attr("fill", q ? "var(--st-ink3)" : "#33475f");
        });
        root.append("text").attr("x", x0 + n * cs / 2).attr("y", y0 - 42).attr("text-anchor", "middle").attr("fill", "var(--st-ink2)").attr("font-size", 12).attr("font-family", "var(--sans)").text("study text of each study");
        studies.forEach((st, i) => {
          const y = y0 + i * cs, g = root.append("g").style("cursor", "pointer").on("click", () => { sel = i; auto = false; tog.sync(); render(); });
          g.append("rect").attr("x", 14).attr("y", y + 1).attr("width", x0 - 22).attr("height", cs - 2).attr("rx", 6).attr("fill", i === sel ? "var(--st-panel2)" : "transparent");
          g.append("text").attr("x", 26).attr("y", y + cs / 2 - 2).attr("fill", "var(--st-ink)").attr("font-size", 13).attr("font-weight", 600).attr("font-family", "var(--sans)").text(st.name);
          g.append("text").attr("x", 26).attr("y", y + cs / 2 + 12).attr("fill", "var(--st-ink3)").attr("font-size", 10.5).attr("font-family", "var(--mono)").text(`${st.views.length} videos: ${st.views.map(shortV).join(", ")}`);
          g.append("rect").attr("x", x0 - 24).attr("y", y + cs / 2 - 7).attr("width", 14).attr("height", 14).attr("rx", 3).attr("fill", col);
          for (let j = 0; j < n; j++) {
            const pos = i === j || same(i, j);
            root.append("rect").attr("x", x0 + j * cs + 1.5).attr("y", y + 1.5).attr("width", cs - 3).attr("height", cs - 3).attr("rx", 3)
              .attr("fill", pos ? col : "var(--st-panel)").attr("fill-opacity", pos && i !== j ? .55 : .95).attr("stroke", "var(--st-line)").attr("opacity", i === sel || j === sel ? 1 : .55);
            if (pos && studies.filter((_, q) => same(i, q) || q === i).length > 1)
              root.append("text").attr("x", x0 + j * cs + cs / 2).attr("y", y + cs / 2 + 4).attr("text-anchor", "middle").attr("fill", "#0d1014").attr("font-size", 12).attr("font-weight", 700).text("½");
          }
        });
        root.append("rect").attr("x", x0 - 2).attr("y", y0 + sel * cs - 2).attr("width", n * cs + 4).attr("height", cs + 4).attr("rx", 5).attr("fill", "none").attr("stroke", "var(--st-ink)").attr("stroke-width", 1.5);
        const lg = root.append("g").attr("transform", `translate(${x0},${y0 + n * cs + 8})`);
        const anySame = texts.some((_, i) => texts.some((_, j) => i !== j && same(i, j)));
        [[col, 1, "positive pair"], ...(anySame ? [[col, .55, "identical text: same positive set"]] : [])].forEach(([f, o, tx], q) => {
          lg.append("rect").attr("y", q * 17).attr("width", 12).attr("height", 12).attr("rx", 2).attr("fill", f).attr("fill-opacity", o);
          lg.append("text").attr("x", 17).attr("y", q * 17 + 10).attr("fill", "var(--st-ink3)").attr("font-size", 11).attr("font-family", "var(--sans)").text(tx);
        });
        const rx = x0 + n * cs + 40, rw = 900 - rx - 10, R = REPORTS[studies[sel].r];
        const fo = root.append("foreignObject").attr("x", rx).attr("y", y0 - 46).attr("width", rw).attr("height", 200);
        const div = fo.append("xhtml:div").attr("style", "font:12px/1.5 var(--sans);color:var(--st-ink2)");
        div.append("xhtml:div").attr("style", "color:var(--st-ink);font-weight:600;margin-bottom:4px").text(`${studies[sel].name}: study text`);
        div.append("xhtml:div").attr("style", "font:11px/1.5 var(--mono);background:var(--st-panel);border:1px solid var(--st-line);border-radius:8px;padding:7px 9px;max-height:150px;overflow:hidden")
          .html(`<b style="color:var(--st-ink)">${R.conc}</b> ` + STRUCT.filter(k => R.f[k]).map(k => `<span style="color:${SCX(k)}">■</span> ${R.f[k]}`).join(" "));
        const k = studies.filter((_, j) => j === sel || same(sel, j)).length;
        const probs = studies.map((_, j) => (j === sel || same(sel, j)) ? (k > 1 ? .5 : 1) : 0);
        targetBars(root.append("g"), rx, y0 + 180, rw, probs, col, "Training target for this row (softmax over the study texts)");
        root.append("foreignObject").attr("x", rx).attr("y", y0 + 236).attr("width", rw).attr("height", 70).append("xhtml:div")
          .attr("style", "font:11px/1.45 var(--sans);color:var(--st-ink3)")
          .text(k > 1 ? "Another study has exactly the same text, so both texts are correct targets: the probability is split equally between them."
                      : "Its text is unique in the batch: the target puts all probability on its own text.");
      }
      render();
      if (!reduceMotion) stackTimer = d3.interval(() => { if (!svg.node().isConnected) { stackTimer && stackTimer.stop(); return; } if (auto) { sel = (sel + 1) % n; render(); } }, 2800);
    }
    // ----- per-structure matrices (one CLIP matrix per cardiac structure k) -----
    function drawStack(svg, kind) {
      const study = kind === "sstruct";
      const short = { "PLAX Standard": "PLAX", "PSAX Papillary": "PSAX papillary", "PSAX Apex": "PSAX apex", "PSAX Pulmonary": "PSAX aortic valve", "SUB IVC": "Subcostal IVC", "Suprasternal Notch": "Suprasternal", "A4C": "A4C", "A2C": "A2C", "A5C": "A5C" };
      const rows = study
        ? [{ name: "Study 1", views: ["A4C", "PLAX Standard"] }, { name: "Study 2", views: ["PSAX Papillary", "PSAX Apex"] }, { name: "Study 3", views: ["SUB IVC"] },
           { name: "Study 4", views: ["A2C", "Suprasternal Notch"] }, { name: "Study 5", views: ["PSAX Pulmonary", "A5C"] }]
        : [{ v: "A4C", img: "assets/camus/a4c_frame.jpg" }, { v: "PLAX Standard" }, { v: "PSAX Papillary" }, { v: "A2C", img: "assets/camus/a2c_frame.jpg" }, { v: "SUB IVC" }, { v: "Suprasternal Notch" }];
      const n = rows.length, cs = study ? 38 : 36, x0 = study ? 360 : 330, y0 = 194, NAME = SNAME;
      const info = k => rows.map(r => {
        if (study) {
          const ws = r.views.map(v => visOf(v)[k]); const anyVis = ws.some(w => w > 0);
          const ok = anyVis;
          return { ok, w: Math.max(...ws), why: ok ? (REPORT[k] ? "" : "no finding → empty-finding text") : `no video shows the ${k}` };
        }
        const w = visOf(r.v)[k], has = !!REPORT[k];
        return { ok: w > 0 && has, w, why: w === 0 ? `${k} not visible in ${short[r.v]} (w = 0)` : !has ? `no ${k} finding in the report` : "" };
      });
      let ai = 0, auto = true;
      const root = svg.append("g");
      // structure chips
      const chips = svg.append("g").attr("transform", "translate(20,56)");
      STRUCT.forEach((k, i) => {
        const g = chips.append("g").attr("transform", `translate(${i * 72},0)`).style("cursor", "pointer").on("click", () => { auto = false; tog.sync(); show(i); });
        g.append("rect").attr("class", "chipbg").attr("width", 66).attr("height", 24).attr("rx", 12).attr("fill", "var(--st-panel)").attr("stroke", "var(--st-line)");
        g.append("rect").attr("x", 9).attr("y", 8).attr("width", 8).attr("height", 8).attr("rx", 2).attr("fill", SCX(k));
        g.append("text").attr("x", 22).attr("y", 16).attr("fill", "var(--st-ink2)").attr("font-size", 11.5).attr("font-family", "var(--mono)").text(k);
      });
      svg.append("text").attr("x", 20).attr("y", 104).attr("fill", "var(--st-ink3)").attr("font-size", 11.5).attr("font-family", "var(--sans)")
        .text(study ? "One matrix per structure k · rows: anatomical study embeddings of k · columns: text of k from each study's report"
                    : "One matrix per structure k · rows: anatomical video embeddings of k · columns: text of k from each video's report");
      const tog = playToggle(svg, 800, 90, () => auto, v => { auto = v; });
      function show(i) {
        ai = i; root.selectAll("*").remove();
        chips.selectAll("g").each(function (_, j) { d3.select(this).select(".chipbg").attr("stroke", j === i ? SCX(STRUCT[i]) : "var(--st-line)").attr("stroke-width", j === i ? 2 : 1).attr("fill", j === i ? "var(--st-panel2)" : "var(--st-panel)"); });
        const k = STRUCT[i], col = SCX(k), inf = info(k), nOk = inf.filter(d => d.ok).length;
        const size = n * cs;
        // stack of the other 11 structure matrices behind (3-D)
        for (let j = 11; j >= 1; j--) {
          const kk = STRUCT[(i + j) % 12];
          root.append("rect").attr("x", x0 + j * 7).attr("y", y0 - j * 5).attr("width", size).attr("height", size).attr("rx", 4)
            .attr("fill", "var(--st-panel)").attr("stroke", SCX(kk)).attr("stroke-opacity", .9 - j * .06).attr("stroke-width", 1.4);
        }
        root.append("text").attr("x", x0).attr("y", y0 + size + 24).attr("fill", "var(--st-ink3)").attr("font-size", 11.5).attr("font-family", "var(--sans)").text("+ 11 more structure matrices behind");
        const front = root.append("g").attr("opacity", 0).attr("transform", "translate(7,-5)");
        front.transition().duration(reduceMotion ? 0 : 320).attr("opacity", 1).attr("transform", "translate(0,0)");
        front.append("rect").attr("x", x0 - 3).attr("y", y0 - 3).attr("width", size + 6).attr("height", size + 6).attr("rx", 5).attr("fill", "var(--st-panel)").attr("stroke", col).attr("stroke-width", 2.2);
        // columns: the structure text of k from each report
        inf.forEach((d, j) => {
          const cx = x0 + j * cs + cs / 2;
          const doc = front.append("g").attr("transform", `translate(${cx - 8},${y0 - 30})`);
          doc.append("path").attr("d", "M0,0 h11 l5,5 v15 h-16 z").attr("fill", "#d9d4c7");
          doc.append("rect").attr("x", 3).attr("y", 8).attr("width", 10).attr("height", 2.5).attr("fill", col);
          doc.append("rect").attr("x", 3).attr("y", 13).attr("width", 8).attr("height", 2).attr("fill", "var(--st-ink3)");
        });
        front.append("text").attr("x", x0 + size / 2).attr("y", y0 - 70).attr("text-anchor", "middle").attr("fill", "var(--st-ink2)").attr("font-size", 12).attr("font-family", "var(--sans)").text(`${k} text from each report ↓`);
        // rows: one per video (or study), with its view and visibility weight for k
        rows.forEach((r, ri) => {
          const y = y0 + ri * cs, d = inf[ri];
          const g = front.append("g");
          // view names only: the site has clips for A4C and A2C alone, so no pictures here
          const label = study ? `${r.name}: ${r.views.map(v => short[v]).join(" + ")}` : short[r.v] + " view";
          g.append("text").attr("x", study ? 20 : 60).attr("y", y + cs / 2 - 2).attr("fill", "var(--st-ink)").attr("font-size", study ? 12 : 13).attr("font-weight", study ? 400 : 600).attr("font-family", "var(--sans)").text(label);
          g.append("text").attr("x", study ? 20 : 60).attr("y", y + cs / 2 + 12).attr("fill", d.w > 0 ? col : "var(--st-ink3)").attr("font-size", 11).attr("font-family", "var(--mono)")
            .text(study ? `max w(${k}) = ${d.w}` : `w(${k}) = ${d.w}`);
          // anatomical embedding token of k for this video/study
          g.append("rect").attr("x", x0 - 22).attr("y", y + cs / 2 - 7).attr("width", 14).attr("height", 14).attr("rx", 3).attr("fill", col).attr("opacity", d.ok ? 1 : .25);
          for (let j = 0; j < n; j++) {
            const diag = ri === j;
            g.append("rect").attr("x", x0 + j * cs + 1.5).attr("y", y + 1.5).attr("width", cs - 3).attr("height", cs - 3).attr("rx", 3)
              .attr("fill", diag ? col : "var(--st-panel)").attr("fill-opacity", diag ? .95 : 1).attr("stroke", "var(--st-line)");
          }
        });
        // mask ineligible rows and columns: animate them being switched off
        inf.forEach((d, ri) => {
          if (d.ok) return;
          const delay = reduceMotion ? 0 : 450 + ri * 120;
          front.append("rect").attr("x", x0).attr("y", y0 + ri * cs).attr("width", size).attr("height", cs).attr("fill", "url(#hatchM)").attr("opacity", 0)
            .transition().delay(delay).duration(350).attr("opacity", .92);
          front.append("rect").attr("x", x0 + ri * cs).attr("y", y0).attr("width", cs).attr("height", size).attr("fill", "url(#hatchM)").attr("opacity", 0)
            .transition().delay(delay).duration(350).attr("opacity", .92);
          front.append("text").attr("x", x0 - 30).attr("y", y0 + ri * cs + cs / 2 + 4).attr("text-anchor", "end").attr("fill", "var(--accent)").attr("font-size", 15).attr("font-weight", 700).text("✕")
            .attr("opacity", 0).transition().delay(delay).duration(350).attr("opacity", 1);
        });
        // right-hand explanation
        const rx = x0 + size + 92, rg = root.append("g").attr("transform", `translate(${rx},150)`);
        rg.append("text").attr("fill", col).attr("font-size", 17).attr("font-weight", 600).attr("font-family", "var(--serif)").text(NAME[k]);
        rg.append("text").attr("y", 24).attr("fill", "var(--st-ink)").attr("font-size", 13).attr("font-family", "var(--sans)")
          .text(`${nOk} of ${n} ${study ? "studies" : "videos"} supervise this matrix`);
        const reasons = inf.map((d, ri) => d.ok ? (study && d.why ? `${rows[ri].name}: ${d.why}` : null) : `${study ? rows[ri].name : short[rows[ri].v]}: ${d.why}`).filter(Boolean);
        const rw = 900 - rx - 8;
        const rfo = rg.append("foreignObject").attr("y", 38).attr("width", rw).attr("height", 150);
        const list = rfo.append("xhtml:div").attr("style", "font:11.5px/1.4 var(--sans);display:flex;flex-direction:column;gap:3px");
        reasons.slice(0, 6).forEach((t, j) => list.append("xhtml:div").attr("style", `color:${t.includes("empty-finding") ? "var(--st-ink3)" : "var(--accent)"};opacity:0;transition:opacity .3s ${reduceMotion ? 0 : .5 + j * .12}s`)
          .text((t.includes("empty-finding") ? "• " : "✕ ") + t));
        requestAnimationFrame(() => requestAnimationFrame(() => list.selectAll("div").style("opacity", 1)));
        const fy = 38 + Math.min(6, reasons.length) * 31 + 10;
        const foot = study ? "A structure is included when at least one video of the study shows it; a missing finding is matched with the encoding of an empty finding."
          : (nOk < 2 ? "Fewer than two eligible videos: this structure is skipped in this batch." : "Eligible: the view has w > 0 and the report has a finding for this structure. The loss is averaged over structures with at least two eligible videos.");
        const fo = rg.append("foreignObject").attr("y", fy).attr("width", rw).attr("height", 120);
        fo.append("xhtml:div").attr("style", "font:11.5px/1.45 var(--sans);color:var(--st-ink3)").text(foot);
      }
      show(0);
      if (!reduceMotion) stackTimer = d3.interval(() => { if (!svg.node().isConnected) { stackTimer && stackTimer.stop(); return; } if (auto) show((ai + 1) % 12); }, 2600);
    }
    draw();
    document.querySelectorAll("#eqline .t").forEach(t => t.addEventListener("click", () => { sel = t.dataset.k; draw(); }));
  }
  overview();
  let lastSz = "";
  const sizeKey = () => stage.clientWidth + "x" + stage.clientHeight;
  onVisible(stage, () => { go(0); lastSz = sizeKey(); });
  const refit = () => { const k = sizeKey(); if (k !== lastSz && stage.clientWidth) { lastSz = k; go(cur); } };
  window.addEventListener("redraw", () => setTimeout(refit, 60));
  let rt; addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(refit, 200); });
})();

/* caption sampler (paper's rule), used in method step 4 */
function captionSampler(host, view0) {
  let view = view0, t = 0.42, inc = [], drop = [], busy = false;
  host.innerHTML = `<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px"><div style="font:600 12px var(--sans)">Visibility weights · <span class="vname"></span></div>
    <button class="btn sm smp" style="background:var(--st-ink);color:var(--st-bg);border-color:var(--st-ink)">⟳ Sample caption</button></div>
    <div class="cch"></div><div class="cst" style="display:flex;gap:6px;flex-wrap:wrap;margin:6px 0"></div>
    <div class="cap" style="font:12px/1.65 var(--mono);background:var(--st-panel);border:1px solid var(--st-line);border-radius:10px;padding:10px 12px;min-height:110px"></div>`;
  const ch = host.querySelector(".cch"), capEl = host.querySelector(".cap"), stEl = host.querySelector(".cst");
  const steps = ["draw threshold", "keep w > t", "drop p = 0.2", "shuffle"];
  stEl.innerHTML = steps.map((s, i) => `<span data-i="${i}" style="font-size:11px;padding:2px 8px;border-radius:99px;border:1px solid var(--st-line);color:var(--st-ink3)">${i + 1} · ${s}</span>`).join("");
  const setStep = k => stEl.querySelectorAll("span").forEach((s, i) => { s.style.background = i <= k ? "var(--st-ink)" : "transparent"; s.style.color = i <= k ? "var(--st-bg)" : "var(--st-ink3)"; });
  let svg, x, y, bars, thr, marks;
  const Wd = 440, Hd = 220, m = { l: 28, r: 6, t: 22, b: 24 };
  function build() {
    ch.innerHTML = ""; host.querySelector(".vname").textContent = view;
    svg = d3.select(ch).append("svg").attr("viewBox", `0 0 ${Wd} ${Hd}`).style("position", "static").style("width", "100%").style("touch-action", "none");
    x = d3.scaleBand().domain(STRUCT).range([m.l, Wd - m.r]).padding(.22); y = d3.scaleLinear().domain([0, 1]).range([Hd - m.b, m.t]);
    svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(4).tickSize(-(Wd - m.l - m.r))).call(g => { g.select(".domain").remove(); g.selectAll("line").attr("stroke", "var(--st-line)"); g.selectAll("text").attr("fill", "var(--st-ink3)").attr("font-size", 10); });
    svg.append("g").attr("transform", `translate(0,${Hd - m.b})`).call(d3.axisBottom(x).tickSize(0)).call(g => { g.select(".domain").remove(); g.selectAll("text").attr("fill", "var(--st-ink2)").attr("font-size", 10).attr("dy", "1.1em"); });
    const w = visOf(view);
    bars = svg.append("g").selectAll("rect").data(STRUCT).join("rect").attr("x", d => x(d)).attr("width", x.bandwidth()).attr("rx", 2).attr("fill", d => SCX(d)).attr("y", d => y(w[d])).attr("height", d => y(0) - y(w[d]));
    marks = svg.append("g");
    thr = svg.append("g"); thr.append("line").attr("x1", m.l).attr("x2", Wd - m.r).attr("stroke", "var(--accent)").attr("stroke-width", 2);
    thr.append("text").attr("x", Wd - m.r).attr("dy", -5).attr("text-anchor", "end").attr("fill", "var(--accent)").attr("font-size", 11).attr("font-weight", 600);
    svg.append("rect").attr("x", m.l).attr("y", m.t).attr("width", Wd - m.l - m.r).attr("height", Hd - m.t - m.b).attr("fill", "transparent").style("cursor", "ns-resize")
      .call(d3.drag().on("start drag", ev => { if (busy) return; const p = d3.pointer(ev, svg.node()); t = Math.max(0, Math.min(.999, y.invert(p[1]))); manual(); }));
    upd(false);
  }
  const W8 = () => visOf(view);
  function upd(anim) {
    const w = W8();
    bars.transition().duration(anim ? 400 : 0).attr("opacity", d => !REPORT[d] ? .15 : (inc.length || drop.length) ? (inc.includes(d) ? 1 : drop.includes(d) ? .45 : .15) : .85);
    thr.transition().duration(anim ? 700 : 0).attr("transform", `translate(0,${y(t)})`); thr.select("text").attr("dy", t > 0.85 ? 14 : -5).text(`t = ${t.toFixed(2)}`);
    marks.selectAll("*").remove();
    drop.forEach(d => marks.append("text").attr("x", x(d) + x.bandwidth() / 2).attr("y", y(w[d]) - 6).attr("text-anchor", "middle").attr("fill", "var(--accent)").attr("font-size", 14).attr("font-weight", 700).text("✕"));
  }
  const prefix = () => `Grayscale B-mode video. ${view} view. Estimated EF 55-60%.`;
  function show(list) {
    capEl.innerHTML = `<span style="color:var(--st-ink3)">${prefix()}</span> ` + (list.length ? list.map((s, i) => `<span style="background:${SCX(s)}33;border-radius:4px;padding:0 3px;animation:fadein .4s ${i * .1}s both">${REPORT[s]}</span>`).join(" ") : `<span style="background:var(--st-line);border-radius:4px;padding:0 3px">${CONCLUSION}</span>`);
  }
  const passing = () => { const w = W8(); return STRUCT.filter(s => REPORT[s] && w[s] > t); };
  function manual() { inc = passing(); drop = []; upd(false); show(inc); setStep(1); }
  const sleep = ms => new Promise(r => setTimeout(r, reduceMotion ? 0 : ms));
  async function sample() {
    if (busy) return; busy = true; inc = []; drop = []; upd(false); show([]); capEl.innerHTML = `<span style="color:var(--st-ink3)">${prefix()}</span>`; setStep(-1);
    t = Math.random(); setStep(0); upd(true); await sleep(800);
    inc = passing(); setStep(1); upd(true); await sleep(600);
    if (inc.length > 1) { drop = inc.filter(() => Math.random() < .2); if (drop.length === inc.length) drop = drop.slice(1); inc = inc.filter(s => !drop.includes(s)); }
    setStep(2); upd(true); await sleep(600);
    setStep(3); show(d3.shuffle(inc.slice())); busy = false;
  }
  host.querySelector(".smp").addEventListener("click", sample);
  build(); setTimeout(sample, 400);
  return { setView(v) { view = v; inc = []; drop = []; build(); sample(); } };
}

/* full visibility matrix (opened in the zoom modal) with a linked 3D heart */
function openMatrix() {
  let H = null;
  Zoom.openCustom("Visibility matrix · 66 views × 12 structures (Extended Data Fig. 1)", host => {
    host.innerHTML = `<div style="display:grid;grid-template-columns:1fr 340px;gap:16px;height:100%"><div class="scroll-x"><div class="chart" id="vmx" style="min-width:820px"></div><div class="note" id="vmxNote" style="margin-top:8px">Hover a column. Weights follow American Society of Echocardiography guidelines.</div></div><div><div class="heart3d" id="vmxHeart"></div><div class="note" id="vmxView" style="margin-top:8px;font-size:14px"></div></div></div>`;
    const el = host.querySelector("#vmx");
    const abbr = { "Apical 2-chamber": "A2C", "Apical 3-chamber": "A3C", "Apical 4-chamber": "A4C", "Apical 5-chamber": "A5C", "Apical Doppler": "Ap. Doppler", "Parasternal long axis": "PLAX", "Long axis Doppler": "LAX Dop.", "Parasternal short axis": "PSAX", "Short axis Doppler": "SAX Dop.", "Subcostal": "Subcostal", "Suprasternal": "SUPA", "Other": "Oth." };
    const W = Math.max(820, el.getBoundingClientRect().width), m = { l: 118, r: 6, t: 46, b: 8 }, cw = (W - m.l - m.r) / VIS.length, chh = 26, Hh = m.t + chh * 12 + m.b;
    const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${W} ${Hh}`);
    d3.groups(VIS.map((v, i) => ({ ...v, i })), d => d.group).forEach(([g, rows]) => {
      const x0 = m.l + rows[0].i * cw, x1 = m.l + (rows.at(-1).i + 1) * cw;
      svg.append("line").attr("x1", x0 + 1).attr("x2", x1 - 1).attr("y1", m.t - 10).attr("y2", m.t - 10).attr("stroke", "var(--ink-3)");
      svg.append("text").attr("x", (x0 + x1) / 2).attr("y", m.t - 16).attr("text-anchor", "middle").attr("fill", "var(--ink-2)").attr("font-size", 11).text(abbr[g] || g);
    });
    STRUCT.forEach((s, j) => { svg.append("rect").attr("x", 4).attr("y", m.t + j * chh + 8).attr("width", 10).attr("height", 10).attr("rx", 2).attr("fill", SCX(s)); svg.append("text").attr("x", 20).attr("y", m.t + j * chh + 17).attr("fill", "var(--ink)").attr("font-size", 12).text(SNAME[s]); });
    const cols = svg.selectAll("g.col").data(VIS).join("g").attr("transform", (d, i) => `translate(${m.l + i * cw},0)`);
    const hl = cols.append("rect").attr("y", m.t - 3).attr("width", cw).attr("height", chh * 12 + 6).attr("rx", 3).attr("fill", "none").attr("stroke", "var(--accent)").attr("stroke-width", 1.6).attr("opacity", 0);
    cols.each(function (d, ci) { d.w.forEach((w, j) => d3.select(this).append("rect").attr("x", .75).attr("y", m.t + j * chh + .75).attr("width", cw - 1.5).attr("height", chh - 1.5).attr("rx", 2).attr("fill", w > 0 ? "var(--m0)" : "var(--card)").attr("fill-opacity", w).attr("stroke", "var(--grid)")); });
    const setV = d => { hl.attr("opacity", x => x === d ? 1 : 0); host.querySelector("#vmxView").innerHTML = `<b>${d.view}</b> · ${d.w.filter(x => x > 0).length} of 12 structures visible`; H && H.setWeights(Object.fromEntries(STRUCT.map((s, k) => [s, d.w[k]]))); };
    cols.append("rect").attr("y", m.t).attr("width", cw).attr("height", chh * 12).attr("fill", "transparent")
      .on("mousemove", function (ev, d) { const j = Math.max(0, Math.min(11, Math.floor((d3.pointer(ev, svg.node())[1] - m.t) / chh))); setV(d); showTip(`<div class="tt">${d.view}</div>${SNAME[STRUCT[j]]}: weight <b>${d.w[j]}</b>`, ev); })
      .on("mouseleave", hideTip);
    whenHeart(() => { H = new Heart3D(host.querySelector("#vmxHeart"), { autoRotate: true }); setV(VIS.find(v => v.view === "A4C")); });
  }, () => { H && H.dispose(); });
}

/* grounding: link heatmap to a 3D heart */
let groundHover = () => {};
(function groundHeart() {
  let H = null;
  onVisible($("#heartGround"), () => whenHeart(() => { H = new Heart3D($("#heartGround"), { autoRotate: true }); }), "200px");
  groundHover = code => {
    if (!H) return;
    const t = $("#grHeartTitle");
    if (!code || code === "PCA") { H.highlight(null); t.textContent = code === "PCA" ? "Combined 832-d embedding: all structures" : "Hover a column"; }
    else { H.highlight(code); t.textContent = `${SNAME[code]} study embedding`; }
  };
})();


/* =========================================================
   RESULTS: data prep
   ========================================================= */
const ZSM = ["EchoAlign", "EchoPrime", "EchoCLIP", "PanEcho"];
const SRCNAME = { private: "Private", mimic: "MIMIC-IV-Echo", cardiobench: "CardioBench" };
const STRUCT_OVR = { rv_overload_septal_motion: "RV", lv_filling_pressure: "LV", lvot_obstruction: "LV", pericardial_effusion: "Peri", cardiac_tamponade: "Peri" };
const TASKS = [];
["private", "mimic"].forEach(src => E.cls.forEach((c, gi) => {
  if (c[src].EchoAlign == null) return;
  TASKS.push({
    id: src + ":" + c.key, src, group: c.group, gi, name: c.name, key: c.key,
    struct: STRUCT_OVR[c.key] || c.structure || "Other",
    lp: c[src], ci: c["ci_" + src],
    zs: c.zs[src], borrowed: c.zs["borrowed_" + src]
  });
}));
E.cardiobench.forEach((c, i) => TASKS.push({
  id: "cb:" + i, src: "cardiobench", group: c.dataset, gi: 100 + i, name: c.task, key: null, struct: c.structure || "Other",
  lp: c.cardiobench, zs: c.zs_cardiobench, borrowed: {}, dataset: c.dataset
}));

/* ---------- leaderboard ---------- */
(function leaderboard() {
  legendModels($("#lbLegend"));
  const sets = [
    { el: "#lb1", models: MODELS, vals: E.means.lp.all, ci: E.means.lp.ci_ea, dom: [0.80, 0.90] },
    { el: "#lb2", models: MODELS, vals: E.means.qwk.all, ci: E.means.qwk.ci_ea, dom: [0.35, 0.55] },
    { el: "#lb3", models: ["EchoAlign", "EchoPrime", "EchoCLIP"], vals: [0.796, 0.692, 0.692], ci: [0.79, 0.81], dom: [0.65, 0.82] }
  ];
  function draw(animate) {
    sets.forEach(S => {
      const el = $(S.el); el.innerHTML = "";
      const rows = S.models.map((m, i) => ({ m, v: S.vals[i] })).sort((a, b) => b.v - a.v);
      const W = width(el), rh = 23, m = { l: 82, r: 54, t: 6, b: 22 }, H = m.t + rh * rows.length + m.b;
      const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${W} ${H}`);
      const x = d3.scaleLinear().domain(S.dom).range([m.l, W - m.r]);
      svg.append("g").attr("class", "grid").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(4).tickSize(-(H - m.b - m.t)).tickFormat(""));
      svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(4).tickSize(0).tickFormat(d3.format(".2f"))).call(g => g.select(".domain").remove()).selectAll("text").attr("dy", "1em");
      const g = svg.selectAll("g.r").data(rows).join("g").attr("class", "r").attr("transform", (d, i) => `translate(0,${m.t + i * rh + rh / 2})`);
      g.append("text").attr("x", m.l - 10).attr("dy", "0.35em").attr("text-anchor", "end").style("fill", d => d.m === "EchoAlign" ? "var(--ink)" : "var(--ink-2)").style("font", d => `${d.m === "EchoAlign" ? 600 : 400} 12.5px var(--sans)`).text(d => d.m);
      g.append("line").attr("x1", m.l).attr("x2", W - m.r).style("stroke", "var(--grid)");
      g.filter(d => d.m === "EchoAlign").append("line").attr("x1", x(S.ci[0])).attr("x2", x(S.ci[1])).style("stroke", "var(--m0)").style("stroke-width", 2).style("opacity", 0)
        .transition().delay(animate ? DUR : 0).duration(300).style("opacity", 0.6);
      g.append("circle").attr("r", d => d.m === "EchoAlign" ? 7 : 5.5).attr("cx", animate ? x(S.dom[0]) : d => x(d.v)).style("fill", d => MC(d.m)).style("stroke", "var(--card)").style("stroke-width", 2)
        .transition().delay((d, i) => animate ? i * 70 : 0).duration(animate ? DUR * 1.4 : 0).ease(d3.easeCubicOut).attr("cx", d => x(d.v));
      g.append("text").attr("x", W - m.r + 12).attr("dy", "0.35em").style("fill", d => d.m === "EchoAlign" ? "var(--ink)" : "var(--ink-2)").style("font", d => `${d.m === "EchoAlign" ? 600 : 400} 12px var(--sans)`).style("font-variant-numeric", "tabular-nums")
        .text(d => fmt3(d.v)).style("opacity", 0).transition().delay(animate ? DUR * 1.2 : 0).duration(300).style("opacity", 1);
    });
  }
  onVisible($("#lb1"), () => draw(true)); onResize(() => draw(false));
})();

/* ---------- task explorer ---------- */
(function explorer() {
  const el = $("#explorer");
  let mode = "strip", ev = "lp", src = "all", sort = "group", vs = "EchoPrime";
  segmented("exMode", v => { mode = v; draw(); });
  segmented("exEval", v => { ev = v; fillVs(); draw(); });
  segmented("exSrc", v => { src = v; draw(); });
  $("#exSort").addEventListener("change", e => { sort = e.target.value; draw(); });
  $("#exVs").addEventListener("change", e => { vs = e.target.value; draw(); });
  function models() { return ev === "lp" ? MODELS : ZSM; }
  function fillVs() {
    const ms = models().slice(1);
    if (!ms.includes(vs)) vs = ms[0];
    $("#exVs").innerHTML = ms.map(m => `<option ${m === vs ? "selected" : ""}>${m}</option>`).join("");
  }
  fillVs();
  function val(t, m) { const v = (ev === "lp" ? t.lp : t.zs)[m]; return v == null ? null : v; }
  function rows() {
    let r = TASKS.filter(t => src === "all" || t.src === src);
    if (ev === "zs") r = r.filter(t => val(t, "EchoAlign") != null);
    const best = t => d3.max(models().slice(1), m => val(t, m) ?? -1);
    const srcOrder = { private: 0, mimic: 1, cardiobench: 2 };
    if (sort === "group") r.sort((a, b) => srcOrder[a.src] - srcOrder[b.src] || a.gi - b.gi);
    else if (sort === "ea") r.sort((a, b) => val(b, "EchoAlign") - val(a, "EchoAlign"));
    else r.sort((a, b) => (val(b, "EchoAlign") - best(b)) - (val(a, "EchoAlign") - best(a)));
    return r;
  }
  function stats(r) {
    const ms = models();
    const ea = d3.mean(r, t => val(t, "EchoAlign"));
    const rank1 = r.filter(t => { const e = val(t, "EchoAlign"); return ms.slice(1).every(m => val(t, m) == null || e >= val(t, m)); }).length;
    const baseMeans = ms.slice(1).filter(m => m !== "PanEcho" || ev === "lp").map(m => [m, d3.mean(r, t => val(t, m))]).filter(d => d[1] != null).sort((a, b) => b[1] - a[1]);
    const paper = ev === "lp" && src === "all";
    const hi = paper ? 31 : r.filter(t => val(t, "EchoAlign") >= 0.9).length;
    $("#exStats").innerHTML = `
      <div class="mini"><div class="n">${fmt3(ea)}</div><div class="l">EchoAlign mean AUROC · ${r.length} tasks</div></div>
      <div class="mini"><div class="n">${baseMeans.length ? fmt3(baseMeans[0][1]) : "–"}</div><div class="l">best other mean (${baseMeans[0]?.[0] || "–"})${ev === "zs" ? ", scored on every task" : ""}</div></div>
      <div class="mini"><div class="n">${rank1}<span style="font-size:16px;color:var(--ink-3)"> / ${r.length}</span></div><div class="l">tasks where EchoAlign ranks first${ev === "zs" ? " (vs EchoPrime, EchoCLIP, PanEcho)" : ""}</div></div>
      <div class="mini"><div class="n">${hi}</div><div class="l">tasks with AUROC ≥ 0.900${paper ? "" : " (rounded values)"}</div></div>`;
    $("#exSub").textContent = `${ev === "lp" ? "Linear probes" : "Zero-shot (text prompts)"} · ${src === "all" ? "all three evaluation sources" : SRCNAME[src]}`;
  }
  const yTicks = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99];
  function tipFor(t, ev2) {
    const ms = models();
    const rs = ms.map(m => [m, val(t, m), ev === "zs" && t.borrowed && t.borrowed[m] ? " *" : ""]).filter(d => d[1] != null).sort((a, b) => b[1] - a[1]);
    return `<div class="tt">${t.name}</div><div class="small muted" style="margin-bottom:4px">${SRCNAME[t.src]}${t.dataset ? " · " + t.dataset : ""} · ${t.src === "cardiobench" ? "" : t.group}</div>${modelTable(rs)}${ev === "zs" && rs.some(r => r[2]) ? `<div class="small muted" style="margin-top:4px">* scored with EchoAlign's/CardioBench's prompts</div>` : ""}${ev === "zs" && val(t, "PanEcho") != null ? `<div class="small muted">PanEcho: supervised head (trained on labels)</div>` : ""}`;
  }
  let lastMode = null;
  function draw() {
    const r = rows(); stats(r);
    $("#exSortWrap").style.display = mode === "strip" ? "" : "none";
    $("#exVsWrap").style.display = mode === "h2h" ? "" : "none";
    const ms = models();
    legendModels($("#exLegend"), mode === "h2h" ? ["EchoAlign", vs] : ms, ev === "zs" ? { PanEcho: " (trained heads)" } : {});
    if (mode !== lastMode) el.innerHTML = "";
    lastMode = mode;
    if (mode === "table") return table(r);
    if (mode === "h2h") return h2h(r);
    strip(r);
  }
  function strip(r) {
    const ms = models();
    const W = Math.max(720, width(el)), m = { l: 44, r: 10, t: 34, b: 16 }, H = isBig(el) ? Math.max(420, innerHeight * .6) : 420;
    let svg = d3.select(el).select("svg");
    if (svg.empty()) {
      el.classList.add("scroll-x");
      svg = d3.select(el).append("svg");
      svg.append("g").attr("class", "grid gy"); svg.append("g").attr("class", "axis ay"); svg.append("g").attr("class", "seps");
      svg.append("g").attr("class", "hov"); svg.append("g").attr("class", "dots"); svg.append("g").attr("class", "hits");
      svg.append("text").attr("class", "axis-title").attr("transform", "rotate(-90)").attr("x", -(H / 2)).attr("y", 12).attr("text-anchor", "middle").text("AUROC (logistic axis)");
    }
    svg.attr("viewBox", `0 0 ${W} ${H}`).style("min-width", "720px");
    const x = d3.scaleBand().domain(r.map(t => t.id)).range([m.l + 8, W - m.r]).padding(0.1);
    const lo = ev === "zs" ? 0.4 : 0.6;
    const y = d3.scaleLinear().domain([logit(lo), logit(0.998)]).range([H - m.b, m.t]);
    const ticks = [lo === 0.4 ? 0.4 : null, ...yTicks].filter(v => v && v >= lo);
    svg.select(".gy").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).tickValues(ticks.map(logit)).tickSize(-(W - m.l - m.r)).tickFormat(""));
    svg.select(".ay").attr("transform", `translate(${m.l},0)`).transition().duration(DUR).call(d3.axisLeft(y).tickValues(ticks.map(logit)).tickSize(0).tickFormat(v => (1 / (1 + Math.exp(-v))).toFixed(2))).call(g => g.select(".domain").remove());
    // group separators (by source and structure group when sorted by group)
    const seps = svg.select(".seps"); seps.selectAll("*").remove();
    if (sort === "group") {
      const blocks = d3.groups(r, t => t.src + "|" + (t.src === "cardiobench" ? "CardioBench" : t.group));
      let prevSrc = null;
      blocks.forEach(([k, ts]) => {
        const [s, g] = k.split("|");
        const x0 = x(ts[0].id) - x.step() * x.padding() / 2, x1 = x(ts.at(-1).id) + x.bandwidth() + x.step() * x.padding() / 2;
        seps.append("line").attr("x1", x0).attr("x2", x0).attr("y1", m.t - (s !== prevSrc ? 30 : 6)).attr("y2", H - m.b).style("stroke", s !== prevSrc ? "var(--ink-3)" : "var(--rule)").style("stroke-width", s !== prevSrc ? 1 : 0.8);
        if (s !== prevSrc && src === "all") seps.append("text").attr("x", x0 + 4).attr("y", m.t - 20).style("fill", "var(--ink)").style("font", "600 11.5px var(--sans)").text(SRCNAME[s]);
        const short = { "Left ventricle": "LV", "Right ventricle": "RV", "Atria": "Atria", "Aortic valve": "AV", "Mitral valve": "MV", "Tricuspid & pulmonic valves": "TV/PV", "Aorta & pulmonary circulation": "Aorta/PA", "Pericardium & IVC": "Peri/IVC", "Devices & acquisition": "Dev." };
        if (s !== "cardiobench" && x1 - x0 > (short[g] || g).length * 6 + 4) seps.append("text").attr("x", (x0 + x1) / 2).attr("y", m.t - 6).attr("text-anchor", "middle").style("fill", "var(--ink-3)").style("font", "10px var(--sans)").text(short[g] || g);
        prevSrc = s;
      });
    }
    // dots
    const pts = [];
    r.forEach(t => ms.forEach(mm => { const v = val(t, mm); if (v != null) pts.push({ t, m: mm, v, k: t.id + mm, hollow: ev === "zs" && t.borrowed && t.borrowed[mm] }); }));
    pts.sort((a, b) => (a.m === "EchoAlign") - (b.m === "EchoAlign"));
    const dots = svg.select(".dots").selectAll("circle").data(pts, d => d.k);
    dots.exit().transition().duration(DUR / 2).attr("r", 0).remove();
    dots.enter().append("circle").attr("cx", d => x(d.t.id) + x.bandwidth() / 2).attr("cy", y(logit(lo))).attr("r", 0)
      .merge(dots)
      .style("fill", d => d.hollow ? "var(--card)" : MC(d.m)).style("stroke", d => d.hollow ? MC(d.m) : (d.m === "EchoAlign" ? "var(--card)" : "none")).style("stroke-width", d => d.hollow ? 1.4 : 1.5)
      .transition().duration(DUR).delay((d, i) => reduceMotion ? 0 : Math.min(400, (x(d.t.id) - m.l) / 3))
      .attr("cx", d => x(d.t.id) + x.bandwidth() / 2).attr("cy", d => y(logit(d.v))).attr("r", d => d.m === "EchoAlign" ? 5 : 3.3);
    // mean lines per model (thin dark ticks at right are noisy; show EA mean as a reference line)
    const ea = d3.mean(r, t => val(t, "EchoAlign"));
    let ml = svg.select(".hov").selectAll("line.mean").data([ea]);
    ml.enter().append("line").attr("class", "mean").merge(ml).transition().duration(DUR)
      .attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(logit(ea))).attr("y2", y(logit(ea))).style("stroke", "var(--m0)").style("stroke-width", 1).style("stroke-dasharray", "4 3").style("opacity", 0.6);
    let mt = svg.select(".hov").selectAll("text.mean").data([ea]);
    mt.enter().append("text").attr("class", "mean").merge(mt).transition().duration(DUR).attr("x", m.l + 4).attr("y", y(logit(ea)) - 5).attr("text-anchor", "start").style("fill", "var(--m0)").style("font", "600 11px var(--sans)").text(`EchoAlign mean ${fmt3(ea)}`);
    // hover columns
    const hl = svg.select(".hov").selectAll("rect.hl").data([0]).join("rect").attr("class", "hl").attr("y", m.t).attr("height", H - m.t - m.b).attr("rx", 3).style("fill", "var(--bg-2)").style("opacity", 0).attr("width", x.step());
    svg.select(".hits").selectAll("rect").data(r, t => t.id).join("rect")
      .attr("x", t => x(t.id) - x.step() * x.padding() / 2).attr("width", x.step()).attr("y", m.t).attr("height", H - m.t - m.b).style("fill", "transparent")
      .on("mousemove", (e, t) => { hl.attr("x", x(t.id) - x.step() * x.padding() / 2).style("opacity", 1); showTip(tipFor(t), e); })
      .on("mouseleave", () => { hl.style("opacity", 0); hideTip(); });
  }
  function h2h(r) {
    const cap = isBig(el) ? Math.min(900, innerHeight * .68) : 620, W = Math.min(width(el), cap), H = W, m = { l: 50, r: 16, t: 16, b: 46 };
    el.classList.remove("scroll-x");
    let svg = d3.select(el).select("svg");
    if (svg.empty()) {
      svg = d3.select(el).append("svg").style("max-width", "620px").style("margin", "0 auto");
      ["grid gx", "grid gy", "axis ax", "axis ay", "diag", "pts", "lab"].forEach(c => svg.append("g").attr("class", c));
      svg.append("text").attr("class", "axis-title xt").attr("text-anchor", "middle");
      svg.append("text").attr("class", "axis-title yt").attr("text-anchor", "middle").attr("transform", "rotate(-90)");
    }
    svg.attr("viewBox", `0 0 ${W} ${H}`).style("max-width", cap + "px");
    const pts = r.map(t => ({ t, a: val(t, "EchoAlign"), b: val(t, vs) })).filter(d => d.a != null && d.b != null);
    const lo = Math.floor(Math.min(d3.min(pts, d => Math.min(d.a, d.b)) ?? 0.6, 0.95) * 20) / 20;
    const x = d3.scaleLinear().domain([lo, 1]).range([m.l, W - m.r]), y = d3.scaleLinear().domain([lo, 1]).range([H - m.b, m.t]);
    svg.select(".gx").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(6).tickSize(-(H - m.t - m.b)).tickFormat(""));
    svg.select(".gy").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(6).tickSize(-(W - m.l - m.r)).tickFormat(""));
    svg.select(".ax").attr("transform", `translate(0,${H - m.b})`).transition().duration(DUR).call(d3.axisBottom(x).ticks(6).tickSize(0).tickFormat(d3.format(".2f"))).selection().selectAll("text").attr("dy", "1.1em");
    svg.select(".ay").attr("transform", `translate(${m.l},0)`).transition().duration(DUR).call(d3.axisLeft(y).ticks(6).tickSize(0).tickFormat(d3.format(".2f")));
    svg.selectAll(".ax .domain,.ay .domain").remove();
    svg.select(".xt").attr("x", (m.l + W - m.r) / 2).attr("y", H - 8).text(`${vs} AUROC`);
    svg.select(".yt").attr("x", -(m.t + H - m.b) / 2).attr("y", 14).text("EchoAlign AUROC");
    const dg = svg.select(".diag"); dg.selectAll("*").remove();
    dg.append("path").attr("d", `M${x(lo)},${y(lo)} L${x(1)},${y(1)} L${x(lo)},${y(1)} Z`).style("fill", "var(--m0)").style("opacity", 0.05);
    dg.append("line").attr("x1", x(lo)).attr("y1", y(lo)).attr("x2", x(1)).attr("y2", y(1)).style("stroke", "var(--ink-3)").style("stroke-dasharray", "4 4");
    const wins = pts.filter(d => d.a > d.b).length;
    dg.append("text").attr("x", x(lo) + 10).attr("y", y(1) + 20).style("fill", "var(--ink)").style("font", "600 13px var(--sans)").text(`EchoAlign higher on ${wins} of ${pts.length}`);
    dg.append("text").attr("x", x(1) - 10).attr("y", y(lo) - 12).attr("text-anchor", "end").style("fill", "var(--ink-2)").style("font", "13px var(--sans)").text(`${vs} higher on ${pts.length - wins}`);
    const c = svg.select(".pts").selectAll("circle").data(pts, d => d.t.id);
    c.exit().transition().duration(DUR / 2).attr("r", 0).remove();
    c.enter().append("circle").attr("r", 0).attr("cx", d => x(d.b)).attr("cy", d => y(d.a)).merge(c)
      .style("fill", d => d.a > d.b ? "var(--m0)" : MC(vs)).style("stroke", "var(--card)").style("stroke-width", 1.5)
      .style("fill-opacity", d => ev === "zs" && d.t.borrowed?.[vs] ? 0.45 : 0.9)
      .on("mousemove", (e, d) => showTip(tipFor(d.t), e)).on("mouseleave", hideTip)
      .transition().duration(DUR).attr("cx", d => x(d.b)).attr("cy", d => y(d.a)).attr("r", 6);
  }
  function table(r) {
    el.classList.remove("scroll-x");
    const ms = models();
    const bySrc = d3.groups(r, t => t.src);
    let h = `<div class="tablewrap"><table class="dtable"><thead><tr><th>Task</th>${ms.map(m => `<th style="color:${m === "EchoAlign" ? "var(--ink)" : ""}">${m}</th>`).join("")}</tr></thead><tbody>`;
    bySrc.forEach(([s, ts]) => {
      h += `<tr class="grp"><td colspan="${ms.length + 1}">${SRCNAME[s]}</td></tr>`;
      ts.forEach(t => {
        const vals = ms.map(m => val(t, m)); const best = d3.max(vals);
        h += `<tr><td title="${t.group}">${t.name}</td>${vals.map((v, i) => `<td class="${v === best ? "best" : ""}">${v == null ? "–" : v.toFixed(3) + (ev === "zs" && t.borrowed?.[ms[i]] ? "*" : "")}</td>`).join("")}</tr>`;
      });
    });
    el.innerHTML = h + "</tbody></table></div>";
  }
  onVisible(el, draw); onResize(() => { lastMode = null; draw(); });
})();

/* ---------- radial: detection across structures ---------- */
(function radial() {
  const el = $("#radial");
  const order = ["LV", "LA", "RV", "RA", "AV", "MV", "TV", "PV", "AoR", "PA", "Peri", "Other"];
  const items = TASKS.map(t => ({ ...t, v: t.lp.EchoAlign })).filter(t => t.v != null);
  const by = order.map(s => ({ s, ts: items.filter(t => t.struct === s).sort((a, b) => a.src.localeCompare(b.src) || b.v - a.v) }));
  function draw() {
    el.innerHTML = "";
    const W = Math.min(isBig(el) ? Math.min(860, innerHeight * .64) : 640, width(el)), H = W, cx = W / 2, cy = H / 2;
    const small = W < 520, R1 = Math.min(W * 0.36, W / 2 - (small ? 74 : 124)), R0 = R1 * 0.55;
    const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${W} ${H}`);
    const r = d3.scaleLinear().domain([0.6, 1]).range([R0, R1]);
    const n = items.length + order.length * 1.4;
    const step = 2 * Math.PI / n;
    let a = 0;
    // reference rings
    [0.7, 0.8, 0.9, 1].forEach(v => svg.append("circle").attr("cx", cx).attr("cy", cy).attr("r", r(v)).style("fill", "none").style("stroke", "var(--grid)").style("stroke-dasharray", v === 1 ? null : "2 3"));
    svg.attr("viewBox", `0 0 ${W} ${H + 40}`);
    const note = svg.append("text").attr("x", cx).attr("y", H + 14).attr("text-anchor", "middle").style("fill", "var(--ink-3)").style("font", "11px var(--sans)");
    note.append("tspan").attr("x", cx).text("Bar length = AUROC of one task");
    note.append("tspan").attr("x", cx).attr("dy", "1.3em").text("inner circle 0.6 · dashed rings 0.7, 0.8, 0.9 · outer circle 1.0");
    const g = svg.append("g").attr("transform", `translate(${cx},${cy})`);
    const sectors = [];
    by.forEach(({ s, ts }) => {
      const a0 = a; a += step * 0.7;
      const sg = g.append("g").attr("class", "sector").style("cursor", "pointer");
      ts.forEach((t, i) => {
        const aa = a + step / 2;
        const arc = d3.arc().innerRadius(R0).outerRadius(r(t.v)).startAngle(a + step * 0.12).endAngle(a + step * 0.88).cornerRadius(1.5);
        sg.append("path").attr("d", arc.outerRadius(R0 + 0.5)()).style("fill", SC(s))
          .style("opacity", t.src === "private" ? 0.95 : t.src === "mimic" ? 0.7 : 0.5)
          .on("mousemove", ev => showTip(`<div class="tt">${t.name}</div>${SRCNAME[t.src]} · EchoAlign AUROC <b>${fmt3(t.v)}</b>`, ev)).on("mouseleave", hideTip)
          .transition().delay(reduceMotion ? 0 : i * 40 + order.indexOf(s) * 60).duration(DUR)
          .attrTween("d", () => { const ir = d3.interpolate(R0 + 0.5, r(t.v)); return k => arc.outerRadius(ir(k))(); });
        a += step;
      });
      a += step * 0.7;
      const a1 = a;
      const mean = (E.structMeans.find(d => d[0] === s) || [0, d3.mean(ts, t => t.v)])[1];
      sg.append("path").attr("d", d3.arc().innerRadius(R1 + 10).outerRadius(R1 + 14).startAngle(a0 + step * 0.4).endAngle(a1 - step * 0.4)()).style("fill", SC(s));
      const mid = (a0 + a1) / 2, lx = Math.sin(mid) * (R1 + (small ? 22 : 34)), ly = -Math.cos(mid) * (R1 + (small ? 22 : 34)) + (Math.cos(mid) < -0.3 ? 8 : 0);
      const anchor = Math.abs(lx) < 20 ? "middle" : lx > 0 ? "start" : "end";
      const tg = sg.append("text").attr("class", "rlab").datum({ lx, ly }).attr("x", lx).attr("y", ly).attr("text-anchor", anchor).style("font", `${small ? 10.5 : 12}px var(--sans)`).style("fill", "var(--ink-2)");
      tg.append("tspan").text(small ? s : SNAME[s].replace(" / global", ""));
      tg.append("tspan").attr("x", lx).attr("dy", "1.2em").style("font-weight", 600).style("fill", "var(--ink)").text(fmt2(mean));
      sg.on("mouseenter", () => { g.selectAll(".sector").style("opacity", function () { return this === sg.node() ? 1 : 0.25; }); $("#radialNote").textContent = `${SNAME[s]}: ${ts.length} tasks, mean AUROC ${fmt3(mean)}`; })
        .on("mouseleave", () => { g.selectAll(".sector").style("opacity", 1); $("#radialNote").textContent = "Hover a structure"; });
      sectors.push(sg);
    });
    // keep structure labels on each side at least two lines apart
    [1, -1].forEach(side => {
      const labs = svg.selectAll(".rlab").nodes().filter(n => Math.sign(d3.select(n).datum().lx || 1) === side).sort((p, q) => d3.select(p).datum().ly - d3.select(q).datum().ly);
      const gap = small ? 26 : 30; let prev = -Infinity;
      labs.forEach(n => {
        const dd = d3.select(n).datum(); let y = dd.ly; if (y < prev + gap) y = prev + gap; prev = y;
        // keep the label outside the ring at its new height (both text lines)
        const R = R1 + (small ? 18 : 30), yy = (y - 12) * (y + 14) < 0 ? 0 : Math.min(Math.abs(y - 12), Math.abs(y + 14)), xr = Math.sqrt(Math.max(0, R * R - Math.min(yy, R) ** 2));
        const x = Math.abs(dd.lx) < 20 ? dd.lx : side * Math.max(Math.abs(dd.lx), xr + 4);
        d3.select(n).attr("y", y).attr("x", x).selectAll("tspan").attr("x", x);
      });
    });
    const cf = Math.max(16, R0 * .5);
    svg.append("text").attr("x", cx).attr("y", cy - 4).attr("text-anchor", "middle").attr("fill", "var(--ink)").attr("font-family", "var(--serif)").attr("font-size", cf).text("0.890");
    if (R0 >= 44) svg.append("text").attr("x", cx).attr("y", cy - 4 + cf * 0.25 + 12).attr("text-anchor", "middle").attr("fill", "var(--ink-3)").attr("font-size", R0 > 60 ? 11 : 10).text("mean AUROC");
  }
  onVisible(el, draw); onResize(draw);
})();

/* =========================================================
   ZERO-SHOT
   ========================================================= */
(function rocChart() {
  const el = $("#roc");
  let band = "lp";
  segmented("rocBand", v => { band = v; draw(false); });
  function draw(anim) {
    el.innerHTML = "";
    const cap = isBig(el) ? Math.min(760, innerHeight * .7) : 420, W = Math.min(width(el), cap), H = W, m = { l: 40, r: 10, t: 10, b: 40 };
    const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${W} ${H}`).style("max-width", cap + "px").style("margin", "0 auto");
    const x = d3.scaleLinear().domain([0, 1]).range([m.l, W - m.r]), y = d3.scaleLinear().domain([0, 1]).range([H - m.b, m.t]);
    svg.append("g").attr("class", "grid").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(5).tickSize(-(H - m.t - m.b)).tickFormat(""));
    svg.append("g").attr("class", "grid").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickSize(-(W - m.l - m.r)).tickFormat(""));
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(5).tickSize(0)).call(g => g.select(".domain").remove()).selectAll("text").attr("dy", "1.1em");
    svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickSize(0)).call(g => g.select(".domain").remove());
    svg.append("text").attr("class", "axis-title").attr("x", (m.l + W) / 2).attr("y", H - 6).attr("text-anchor", "middle").text("1 − specificity");
    svg.append("text").attr("class", "axis-title").attr("transform", "rotate(-90)").attr("x", -(H - m.b) / 2).attr("y", 11).attr("text-anchor", "middle").text("Sensitivity");
    svg.append("line").attr("x1", x(0)).attr("y1", y(0)).attr("x2", x(1)).attr("y2", y(1)).style("stroke", "var(--rule)").style("stroke-dasharray", "3 3");
    const line = d3.line().x(d => x(d[0])).y(d => y(d[1]));
    const b = E.roc[band === "lp" ? "band_linear" : "band_text"];
    svg.append("path").attr("d", line(b) + "Z").style("fill", "var(--m0)").style("opacity", 0).transition().duration(DUR).style("opacity", 0.12);
    const draw1 = (pts, dash) => {
      const p = svg.append("path").attr("d", line(pts)).style("fill", "none").style("stroke", "var(--m0)").style("stroke-width", 2.4);
      if (anim && !reduceMotion) {
        const L = p.node().getTotalLength();
        p.attr("stroke-dasharray", `${L} ${L}`).attr("stroke-dashoffset", L).transition().duration(1400).ease(d3.easeCubicOut).attr("stroke-dashoffset", 0)
          .on("end", () => p.attr("stroke-dasharray", dash || null));
      } else p.attr("stroke-dasharray", dash || null);
    };
    draw1(E.roc.linear, null); draw1(E.roc.text, "7 4");
  }
  onVisible(el, () => draw(true)); onResize(() => draw(false));
})();

function ridgeChart(el, curves, xdom, xticks, xfmt, order, opt = {}) {
  el.innerHTML = "";
  const W = width(el), H = (opt.h || 250) * (isBig(el) ? 1.9 : 1), m = { l: 12, r: 12, t: 10, b: 34 };
  const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${W} ${H}`);
  const x = d3.scaleLinear().domain(xdom).range([m.l, W - m.r]);
  const y = d3.scaleLinear().domain([0, 1]).range([H - m.b, m.t]);
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).tickValues(xticks).tickSize(4).tickFormat(xfmt)).call(g => g.select(".domain").style("stroke", "var(--rule)")).selectAll("text").attr("dy", "1em");
  if (opt.xtitle) svg.append("text").attr("class", "axis-title").attr("x", (m.l + W - m.r) / 2).attr("y", H - 2).attr("text-anchor", "middle").text(opt.xtitle);
  const area = d3.area().x(d => x(d[0])).y0(y(0)).y1(d => y(d[1])).curve(d3.curveBasis);
  const line = d3.line().x(d => x(d[0])).y(d => y(d[1])).curve(d3.curveBasis);
  order.forEach((gname, i) => {
    const pts = curves[gname]; if (!pts) return;
    const gi = GRADES.indexOf(gname);
    const flat = pts.map(p => [p[0], 0]);
    const g = svg.append("g").style("cursor", "default");
    const a = g.append("path").attr("d", area(flat)).style("fill", GC(gi)).style("fill-opacity", 0.28);
    const l = g.append("path").attr("d", line(flat)).style("fill", "none").style("stroke", GC(gi)).style("stroke-width", 2);
    const delay = reduceMotion ? 0 : i * 260;
    a.transition().delay(delay).duration(DUR).attr("d", area(pts));
    l.transition().delay(delay).duration(DUR).attr("d", line(pts));
    // median-ish marker: x at curve peak
    const peak = pts.reduce((a, b) => b[1] > a[1] ? b : a);
    g.append("line").attr("x1", x(peak[0])).attr("x2", x(peak[0])).attr("y1", y(0)).attr("y2", y(0) + 6).style("stroke", GC(gi)).style("stroke-width", 2.5).style("opacity", 0).transition().delay(delay + DUR).style("opacity", 1);
    g.on("mousemove", ev => showTip(`<div class="tt">${gname}</div>Distribution peak at ${xfmt(peak[0])}`, ev)).on("mouseleave", hideTip)
      .on("mouseenter", () => svg.selectAll("g").style("opacity", function () { return this === g.node() || !this.parentNode || this.parentNode !== svg.node() ? 1 : 0.35; }))
      .on("mouseout", () => svg.selectAll("g").style("opacity", 1));
  });
  return svg;
}
function gradeLegend(el, names) { el.innerHTML = names.map(n => `<span class="k"><span class="sw" style="background:${GC(GRADES.indexOf(n))}"></span>${n}</span>`).join(""); }

(function ridges() {
  const el = $("#ridge"), sel = $("#ridgeSel");
  const names = Object.keys(E.textByGrade);
  let cur = names[0];
  sel.innerHTML = names.map((n, i) => `<button data-v="${n}" class="${i ? "" : "on"}">${n}</button>`).join("");
  segmented("ridgeSel", v => { cur = v; draw(); });
  function draw() {
    const d = E.textByGrade[cur], curves = d.curves;
    const xs = Object.values(curves).flat().map(p => p[0]);
    const order = GRADES.filter(g => curves[g]);
    const ticks = d.ticks.map(t => t[0]);
    const svg = ridgeChart(el, curves, [d3.min(xs), d3.max(xs)], ticks, d3.format(".1f"), order, { xtitle: "Zero-shot score", h: 230 });
    const rho = E.spearman[cur];
    svg.append("text").attr("x", 14).attr("y", 22).style("fill", "var(--ink)").style("font", "600 13px var(--sans)").text(`Spearman ρ = ${rho[0].toFixed(2)} (${rho[1].toFixed(2)}–${rho[2].toFixed(2)})`);
    gradeLegend($("#gradeLegend"), order);
  }
  onVisible(el, draw); onResize(draw);
})();

(function promptInspector() {
  const sel = $("#piSel");
  const list = E.cls.filter(c => c.prompts);
  sel.innerHTML = d3.groups(list, c => c.group).map(([g, cs]) => `<optgroup label="${g}">${cs.map(c => `<option value="${c.key}" ${c.key === "tricuspid_regurgitation" ? "selected" : ""}>${c.name}</option>`).join("")}</optgroup>`).join("");
  let src = "private";
  segmented("piSrc", v => { src = v; draw(); });
  sel.addEventListener("change", draw);
  const el = $("#piBars");
  function draw() {
    const c = list.find(x => x.key === sel.value);
    const s = c.structure;
    $("#piStruct").innerHTML = s ? `<span class="pill" style="border-color:${SC(s)};color:var(--ink)"><span style="display:inline-block;width:9px;height:9px;border-radius:2px;background:${SC(s)};margin-right:6px"></span>${SNAME[s]} study embedding</span>` : `<span class="pill">Global study embedding</span>`;
    $("#piPos").innerHTML = c.prompts.positive.map(p => `<div class="prompt pos">${p}</div>`).join("");
    $("#piNeg").innerHTML = c.prompts.negative.map(p => `<div class="prompt neg">${p}</div>`).join("");
    const z = c.zs[src], br = c.zs["borrowed_" + src];
    el.innerHTML = "";
    if (z.EchoAlign == null) { el.innerHTML = `<div class="muted small" style="padding:30px 0">Not a task in ${SRCNAME[src]}.</div>`; $("#piNote").textContent = ""; return; }
    const rows = ZSM.map(m => ({ m, v: z[m], b: br[m] })).filter(d => d.v != null);
    const W = width(el), rh = 40, mg = { l: 86, r: 50, t: 8, b: 28 }, H = mg.t + rows.length * rh + mg.b;
    const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${W} ${H}`);
    const x = d3.scaleLinear().domain([0.4, 1]).range([mg.l, W - mg.r]);
    svg.append("g").attr("class", "grid").attr("transform", `translate(0,${H - mg.b})`).call(d3.axisBottom(x).ticks(6).tickSize(-(H - mg.t - mg.b)).tickFormat(""));
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - mg.b})`).call(d3.axisBottom(x).ticks(6).tickSize(0).tickFormat(d3.format(".1f"))).call(g => g.select(".domain").remove()).selectAll("text").attr("dy", "1em");
    svg.append("line").attr("x1", x(0.5)).attr("x2", x(0.5)).attr("y1", mg.t).attr("y2", H - mg.b).style("stroke", "var(--ink-3)").style("stroke-dasharray", "3 3");
    svg.append("text").attr("x", x(0.5) + 4).attr("y", mg.t + 10).style("fill", "var(--ink-3)").style("font", "10.5px var(--sans)").text("chance");
    const g = svg.selectAll("g.r").data(rows).join("g").attr("class", "r").attr("transform", (d, i) => `translate(0,${mg.t + i * rh + rh / 2})`);
    g.append("text").attr("x", mg.l - 10).attr("dy", "0.35em").attr("text-anchor", "end").style("font", d => `${d.m === "EchoAlign" ? 600 : 400} 12.5px var(--sans)`).style("fill", "var(--ink)").text(d => d.m + (d.b ? "*" : ""));
    g.append("rect").attr("x", x(0.4)).attr("y", -7).attr("height", 14).attr("rx", 4).attr("width", 0)
      .style("fill", d => d.b ? "url(#piHatch)" : MC(d.m)).style("stroke", d => MC(d.m)).style("stroke-width", d => d.b ? 1.2 : 0)
      .transition().duration(DUR).attr("width", d => x(d.v) - x(0.4));
    g.append("text").attr("x", d => x(d.v) + 6).attr("dy", "0.35em").style("font", "600 12px var(--sans)").style("fill", "var(--ink)").style("font-variant-numeric", "tabular-nums").text(d => fmt3(d.v));
    const defs = svg.append("defs").append("pattern").attr("id", "piHatch").attr("width", 5).attr("height", 5).attr("patternUnits", "userSpaceOnUse").attr("patternTransform", "rotate(45)");
    defs.append("rect").attr("width", 5).attr("height", 5).style("fill", "var(--card)");
    defs.append("line").attr("y2", 5).style("stroke", "var(--ink-3)").style("stroke-width", 2);
    $("#piNote").innerHTML = (rows.some(r => r.b) ? "* Hatched: the model's own method does not cover this task, so it is scored with EchoAlign's prompts encoded by its own text encoder. " : "") + (z.PanEcho != null ? "PanEcho uses a supervised head trained on labels." : "");
  }
  onVisible(el, draw); onResize(draw);
})();

/* =========================================================
   SEVERITY GRADING
   ========================================================= */
(function expectedGrade() {
  const names = Object.keys(E.expectedGrade);
  let cur = names[0];
  $("#egSel").innerHTML = names.map((n, i) => `<button data-v="${n}" class="${i ? "" : "on"}">${n.split(" ")[0]}</button>`).join("");
  segmented("egSel", v => { cur = v; draw(); });
  function draw() {
    const d = E.expectedGrade[cur];
    const order = GRADES.filter(g => d.curves[g]);
    ridgeChart($("#egRidge"), d.curves, [-0.15, 4.15], [0, 1, 2, 3, 4], v => ["Normal", "Trace", "Mild", "Moderate", "Severe"][v], order, { xtitle: "Expected grade", h: 230 });
    gradeLegend($("#egLegend"), order);
    const s = E.sevAUC[cur];
    // ROC
    const el = $("#egRoc"); el.innerHTML = "";
    const cap = isBig(el) ? 520 : 300, W = Math.min(Math.max(200, el.getBoundingClientRect().width || cap), cap), H = W, m = { l: 36, r: 8, t: 8, b: 34 };
    const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${W} ${H}`).style("max-width", cap + "px");
    const x = d3.scaleLinear().domain([0, 1]).range([m.l, W - m.r]), y = d3.scaleLinear().domain([0, 1]).range([H - m.b, m.t]);
    svg.append("g").attr("class", "grid").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(4).tickSize(-(H - m.t - m.b)).tickFormat(""));
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(4).tickSize(0)).call(g => g.select(".domain").remove()).selectAll("text").attr("dy", "1em");
    svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(4).tickSize(0)).call(g => g.select(".domain").remove());
    svg.append("text").attr("class", "axis-title").attr("x", (m.l + W) / 2).attr("y", H - 3).attr("text-anchor", "middle").text("1 − specificity");
    svg.append("line").attr("x1", x(0)).attr("y1", y(0)).attr("x2", x(1)).attr("y2", y(1)).style("stroke", "var(--rule)").style("stroke-dasharray", "3 3");
    const line = d3.line().x(p => x(p[0])).y(p => y(p[1]));
    [["roc_moderate", "var(--g3)"], ["roc_severe", "var(--g4)"]].forEach(([k, c], i) => {
      const p = svg.append("path").attr("d", line(d[k])).style("fill", "none").style("stroke", c).style("stroke-width", 2.2);
      if (!reduceMotion) { const L = p.node().getTotalLength(); p.attr("stroke-dasharray", `${L} ${L}`).attr("stroke-dashoffset", L).transition().delay(i * 300).duration(1100).attr("stroke-dashoffset", 0); }
    });
    $("#egStats").innerHTML = `<div class="legend" style="margin-bottom:4px"><span class="k"><span class="ln" style="border-color:var(--g3)"></span>Moderate or worse: <b>&nbsp;${s.mod[0].toFixed(2)}</b>&nbsp;(${s.mod[1].toFixed(2)}–${s.mod[2].toFixed(2)}), ${s.mod[3]} studies</span></div>
      <div class="legend" style="margin-bottom:4px"><span class="k"><span class="ln" style="border-color:var(--g4)"></span>Severe: <b>&nbsp;${s.sev[0].toFixed(2)}</b>&nbsp;(${s.sev[1].toFixed(2)}–${s.sev[2].toFixed(2)}), ${s.sev[3]} studies</span></div>
      <div class="muted">QWK ${s.qwk[0].toFixed(2)} (${s.qwk[1].toFixed(2)}–${s.qwk[2].toFixed(2)})</div>`;
  }
  onVisible($("#egRidge"), draw); onResize(draw);
})();

function dotRows(el, rows, models, dom, opt = {}) {
  // rows: [{label, sub?, vals:{model:v}, header?}]
  el.innerHTML = "";
  const W = width(el), rh = opt.rh || 24, m = { l: opt.ml || 210, r: 50, t: 6, b: 28 };
  if (W < 560) m.l = Math.min(m.l, 150);
  const H = m.t + rows.length * rh + m.b;
  const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${W} ${H}`);
  const x = d3.scaleLinear().domain(dom).range([m.l, W - m.r]);
  svg.append("g").attr("class", "grid").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(W < 560 ? 3 : 6).tickSize(-(H - m.t - m.b)).tickFormat(""));
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(W < 560 ? 3 : 6).tickSize(0).tickFormat(opt.fmt || d3.format(".2f"))).call(g => g.select(".domain").remove()).selectAll("text").attr("dy", "1em");
  const g = svg.selectAll("g.r").data(rows).join("g").attr("class", "r").attr("transform", (d, i) => `translate(0,${m.t + i * rh + rh / 2})`);
  g.filter(d => d.header).append("text").attr("x", 4).attr("dy", "0.35em").style("font", "600 12px var(--sans)").style("fill", "var(--ink)").text(d => d.label);
  const rr = g.filter(d => !d.header);
  rr.append("rect").attr("x", 0).attr("y", -rh / 2).attr("width", W).attr("height", rh).style("fill", "transparent")
    .on("mousemove", (ev, d) => showTip(`<div class="tt">${d.label}</div>${modelTable(models.map(mm => [mm, d.vals[mm]]).filter(r => r[1] != null).sort((a, b) => opt.lowerBetter ? a[1] - b[1] : b[1] - a[1]), opt.digits || 3)}${d.note ? `<div class="small muted" style="margin-top:4px">${d.note}</div>` : ""}`, ev))
    .on("mouseleave", hideTip);
  const maxChars = Math.floor((m.l - 12) / 6.2);
  rr.append("text").attr("x", m.l - 10).attr("dy", "0.35em").attr("text-anchor", "end").style("font", "12px var(--sans)").style("fill", "var(--ink-2)")
    .text(d => d.label.length > maxChars ? d.label.slice(0, maxChars - 1) + "…" : d.label).style("pointer-events", "none");
  rr.append("line").attr("x1", d => x(d3.min(models, mm => d.vals[mm] ?? Infinity))).attr("x2", d => x(d3.max(models, mm => d.vals[mm] ?? -Infinity))).style("stroke", "var(--rule)").style("stroke-width", 2).style("pointer-events", "none");
  models.slice().reverse().forEach(mm => {
    rr.filter(d => d.vals[mm] != null).append("circle").attr("cx", x(dom[0])).attr("r", mm === "EchoAlign" ? 5.5 : 3.8)
      .style("fill", MC(mm)).style("stroke", mm === "EchoAlign" ? "var(--card)" : "none").style("stroke-width", 1.5).style("pointer-events", "none")
      .transition().duration(DUR).delay((d, i) => reduceMotion ? 0 : i * 25).attr("cx", d => x(Math.max(dom[0], Math.min(dom[1], d.vals[mm]))));
  });
  if (opt.ref != null) svg.append("line").attr("x1", x(opt.ref)).attr("x2", x(opt.ref)).attr("y1", m.t).attr("y2", H - m.b).style("stroke", "var(--ink-3)").style("stroke-dasharray", "3 3");
  return svg;
}

(function qwk() {
  const el = $("#qwk"); let src = "private";
  legendModels($("#qwkLegend"));
  segmented("qwkSrc", v => { src = v; draw(); });
  function draw() {
    const rows = E.qwk.filter(q => q[src].EchoAlign != null).map(q => ({ label: q.group === "Left ventricle" || q.group === "Right ventricle" ? q.group.replace("ventricle", "ventr.") + " " + q.task.toLowerCase() : (q.group === "Aortic valve" || q.group === "Mitral valve") ? q.group.split(" ")[0] + " " + q.task.toLowerCase() : q.task, vals: q[src] }));
    const mean = src === "private" ? [0.473, 0.403, 0.376, 0.369, 0.340, 0.301] : [0.562, 0.561, 0.501, 0.536, 0.497, 0.437];
    rows.push({ label: "Mean of all findings", vals: Object.fromEntries(MODELS.map((m, i) => [m, mean[i]])) });
    dotRows(el, rows, MODELS, [0, 0.9]);
  }
  onVisible(el, draw); onResize(draw);
})();

/* =========================================================
   SEGMENTATION + MEASUREMENT
   ========================================================= */
const RMS_SHORT = { "Ejection fraction": "Ejection fraction", "End-diastolic internal diameter": "LV diastolic diameter", "End-systolic internal diameter": "LV systolic diameter", "Resting outflow tract pressure gradient": "LVOT gradient", "Left atrial volume index": "LA volume index", "Left atrial volume": "LA volume", "Mitral inflow E/A ratio": "Mitral E/A ratio", "Pulmonary artery pressure": "PA pressure", "Estimated right ventricular systolic pressure": "RV systolic pressure", "Tricuspid regurgitation peak velocity": "TR peak velocity", "TAPSE": "TAPSE", "Aortic root diameter": "Aortic root diameter", "Mean transaortic pressure gradient": "Mean AV gradient", "Peak transaortic pressure gradient": "Peak AV gradient", "Peak transaortic velocity": "Peak AV velocity" };
(function seg() {
  legendModels($("#bLegend"));
  const wrap = $("#segImgs");
  const imgs = ["camus_good", "camus_medium", "camus_poor"], labels = ["Good quality", "Medium quality", "Poor quality"];
  wrap.innerHTML = imgs.map((s, i) => `<div><div style="position:relative;border-radius:12px;overflow:hidden;background:#000;aspect-ratio:129/89"><img src="assets/${s}.png" alt="CAMUS ${labels[i]} image" style="width:100%;height:100%;display:block;image-rendering:auto"><svg viewBox="0 0 129 89" preserveAspectRatio="none" style="position:absolute;inset:0;width:100%;height:100%" data-i="${i}"></svg></div><div class="small muted" style="margin-top:6px;text-align:center">${labels[i]}</div></div>`).join("");
  function drawContours() {
    wrap.querySelectorAll("svg").forEach(svgEl => {
      const i = +svgEl.dataset.i, c = E.seg.contours[i];
      const svg = d3.select(svgEl); svg.selectAll("*").remove();
      const line = d3.line().x(p => p[0] * 129).y(p => p[1] * 89).curve(d3.curveLinearClosed);
      c.echoalign.forEach((pts, k) => {
        const p = svg.append("path").attr("d", line(pts)).style("fill", "#5f95dc").style("fill-opacity", 0).style("stroke", "#8dbaf7").style("stroke-width", 0.9).style("stroke-linejoin", "round");
        const L = p.node().getTotalLength();
        if (reduceMotion) { p.style("fill-opacity", 0.16); return; }
        p.attr("stroke-dasharray", `${L} ${L}`).attr("stroke-dashoffset", L)
          .transition().delay(k * 260).duration(1200).ease(d3.easeCubicInOut).attr("stroke-dashoffset", 0)
          .transition().duration(500).style("fill-opacity", 0.16);
      });
      c.expert.forEach((pts, k) => {
        const p = svg.append("path").attr("d", line(pts)).style("fill", "none").style("stroke", "#fff").style("stroke-width", 0.7).attr("stroke-dasharray", "2.2 1.4");
        p.style("opacity", 0).transition().delay(reduceMotion ? 0 : 1300 + k * 200).duration(600).style("opacity", 0.95);
      });
    });
  }
  onVisible(wrap, drawContours);
  $("#segReplay").addEventListener("click", drawContours);
  let mode = "same";
  segmented("segMode", v => { mode = v; dice(); });
  function dice() {
    const short = n => n.replace("EchoNet-Dynamic", "EchoNet-Dyn.").replace("EchoNet-Pediatric", "EchoNet-Ped.");
    const rows = E.seg[mode].map(r => ({ label: short(r.name), vals: Object.fromEntries(MODELS.map((m, i) => [m, r.v[i]])) }));
    dotRows($("#segDice"), rows, MODELS, mode === "same" ? [0.7, 0.95] : [0.5, 0.95], { ml: 190, rh: 28 });
  }
  onVisible($("#segDice"), dice); onResize(dice);
  let rsrc = "private";
  segmented("rmSrc", v => { rsrc = v; rmse(); });
  function rmse() {
    let rows;
    if (rsrc === "cb") rows = E.cb_rmse.map(r => ({ label: `${r.task.replace("Ejection fraction", "EF").replace(" (diastole)", "")} · ${r.dataset.replace("EchoNet-", "EN-")}`, raw: Object.fromEntries(MODELS.map((m, i) => [m, r.v[i]])), unit: r.unit }));
    else rows = E.rmse.filter(r => r[rsrc].EchoAlign != null).map(r => ({ label: RMS_SHORT[r.task.replace("circulation ", "")] || r.task, raw: r[rsrc] }));
    rows.forEach(r => {
      const best = d3.min(MODELS, m => r.raw[m]);
      r.vals = Object.fromEntries(MODELS.map(m => [m, r.raw[m] / best]));
      r.note = "Raw RMSE: " + MODELS.map(m => `${m} ${r.raw[m]}`).join(" · ");
    });
    const mx = Math.min(1.6, d3.max(rows, r => d3.max(MODELS, m => r.vals[m])));
    dotRows($("#rmse"), rows, MODELS, [1, Math.max(1.25, mx)], { ml: 200, rh: 26, lowerBetter: true, digits: 2, fmt: v => "×" + v.toFixed(2), ref: 1 });
  }
  onVisible($("#rmse"), rmse); onResize(rmse);
})();

/* =========================================================
   EFFICIENCY
   ========================================================= */
(function scaling() {
  const el = $("#scaling");
  function draw(anim) {
    el.innerHTML = "";
    const W = width(el), H = isBig(el) ? Math.min(innerHeight * .66, W * .6) : Math.min(340, W * 0.8), m = { l: 44, r: 16, t: 14, b: 42 };
    const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${W} ${H}`);
    const x = d3.scaleLog().domain([2500, 350000]).range([m.l, W - m.r]), y = d3.scaleLinear().domain([0.70, 0.91]).range([H - m.b, m.t]);
    const tv = [3000, 10000, 30000, 100000, 300000];
    svg.append("g").attr("class", "grid").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickSize(-(W - m.l - m.r)).tickFormat(""));
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).tickValues(tv).tickSize(0).tickFormat(d => d >= 1000 ? d / 1000 + "k" : d)).call(g => g.select(".domain").style("stroke", "var(--rule)")).selectAll("text").attr("dy", "1.1em");
    svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickSize(0).tickFormat(d3.format(".2f"))).call(g => g.select(".domain").remove());
    svg.append("text").attr("class", "axis-title").attr("x", (m.l + W) / 2).attr("y", H - 4).attr("text-anchor", "middle").text("Pretraining studies (log scale)");
    const S = E.scaling;
    // gap at 30k studies: drawn first so the dots sit on its ends
    const i30 = S.subsets.indexOf(30000);
    if (i30 >= 0) {
      const ya = y(S.EchoAlign[i30]), yp = y(S.EchoPrime[i30]), xx = x(30000);
      svg.append("line").attr("x1", xx).attr("x2", xx).attr("y1", ya).attr("y2", yp).style("stroke", "var(--accent)").style("stroke-width", 2.5);
      svg.append("text").attr("x", xx + 11).attr("y", (ya + yp) / 2 + 4).style("fill", "var(--accent)").style("font", "700 13px var(--sans)").text("+0.067"); // paper value; the rounded points give 0.066
    }
    // y of a series' line at screen x (straight segments in log-x space)
    const lineY = (k, px) => { const xs = S.subsets.map(x), v = S[k]; let j = d3.bisectRight(xs, px) - 1; j = Math.max(0, Math.min(xs.length - 2, j)); const t = Math.max(0, Math.min(1, (px - xs[j]) / (xs[j + 1] - xs[j]))); return y(v[j] + t * (v[j + 1] - v[j])); };
    [["EchoPrime", "var(--m1)"], ["EchoAlign", "var(--m0)"]].forEach(([k, c], j) => {
      const pts = S.subsets.map((s, i) => [s, S[k][i]]);
      const p = svg.append("path").attr("d", d3.line().x(d => x(d[0])).y(d => y(d[1]))(pts)).style("fill", "none").style("stroke", c).style("stroke-width", 2.2);
      if (anim && !reduceMotion) { const L = p.node().getTotalLength(); p.attr("stroke-dasharray", `${L} ${L}`).attr("stroke-dashoffset", L).transition().delay(j * 300).duration(1200).attr("stroke-dashoffset", 0); }
      svg.selectAll(null).data(pts).join("circle").attr("cx", d => x(d[0])).attr("cy", d => y(d[1])).attr("r", 5.5).style("fill", "var(--card)").style("stroke", c).style("stroke-width", 2.2)
        .on("mousemove", (ev, d) => showTip(`<div class="tt">${k} recipe · ${d3.format(",")(d[0])} studies</div>Mean AUROC <b>${fmt3(d[1])}</b>`, ev)).on("mouseleave", hideTip);
      const [n, v] = S.released[k];
      svg.append("circle").attr("cx", x(n)).attr("cy", y(v)).attr("r", 7).style("fill", c).style("stroke", "var(--card)").style("stroke-width", 2)
        .on("mousemove", ev => showTip(`<div class="tt">Released ${k}</div>${d3.format(",")(n)} studies · mean AUROC <b>${fmt3(v)}</b>${k === "EchoPrime" ? "<div class='small muted'>trained on a different archive</div>" : ""}`, ev)).on("mouseleave", hideTip);
      svg.append("text").attr("x", k === "EchoAlign" ? x(n) - 10 : Math.min(x(n) + 8, W - m.r)).attr("y", k === "EchoAlign" ? y(v) + 4 : y(v) + 22).attr("text-anchor", "end").style("font", "600 11.5px var(--sans)").style("fill", "var(--ink)").text(`released ${fmt3(v)}`);
      // series label: above the EchoAlign line, below the EchoPrime line, clear of the line along its whole width
      const x1 = x(S.subsets[0]) + 10, x2 = x1 + 100, above = k === "EchoAlign";
      const ys = d3.range(0, 11).map(q => lineY(k, x1 + (x2 - x1) * q / 10));
      svg.append("text").attr("x", x1).attr("y", above ? d3.min(ys) - 10 : d3.max(ys) + 20).style("font", "600 11px var(--sans)").style("fill", c).text(`${k} recipe`);
    });
  }
  onVisible(el, () => draw(true)); onResize(() => draw(false));
})();

(function labelEff() {
  const el = $("#labeleff"); let src = "private";
  legendModels($("#leLegend"));
  segmented("leSrc", v => { src = v; draw(false); });
  function draw(anim) {
    el.innerHTML = "";
    const W = width(el), H = isBig(el) ? Math.min(innerHeight * .66, W * .6) : Math.min(340, W * 0.8), m = { l: 44, r: 70, t: 14, b: 42 };
    const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${W} ${H}`);
    const F = E.labeleff.fractions, D = E.labeleff[src];
    const x = d3.scaleLog().domain([1, 100]).range([m.l, W - m.r]);
    const ys = Object.values(D).flat();
    const y = d3.scaleLinear().domain([Math.floor(d3.min(ys) * 50) / 50, Math.ceil(d3.max(ys) * 50) / 50]).range([H - m.b, m.t]);
    svg.append("g").attr("class", "grid").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickSize(-(W - m.l - m.r)).tickFormat(""));
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).tickValues(F).tickSize(0).tickFormat(d => d + "%")).call(g => g.select(".domain").style("stroke", "var(--rule)")).selectAll("text").attr("dy", "1.1em");
    svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickSize(0).tickFormat(d3.format(".2f"))).call(g => g.select(".domain").remove());
    svg.append("text").attr("class", "axis-title").attr("x", (m.l + W - m.r) / 2).attr("y", H - 4).attr("text-anchor", "middle").text("Labels used for linear probing (log scale)");
    const line = d3.line().x((d, i) => x(F[i])).y(d => y(d));
    MODELS.slice().reverse().forEach(mm => {
      const p = svg.append("path").attr("d", line(D[mm])).style("fill", "none").style("stroke", MC(mm)).style("stroke-width", mm === "EchoAlign" ? 2.8 : 1.6);
      if (anim && !reduceMotion) { const L = p.node().getTotalLength(); p.attr("stroke-dasharray", `${L} ${L}`).attr("stroke-dashoffset", L).transition().duration(1300).attr("stroke-dashoffset", 0); }
      svg.selectAll(null).data(D[mm]).join("circle").attr("cx", (d, i) => x(F[i])).attr("cy", d => y(d)).attr("r", mm === "EchoAlign" ? 3.6 : 2.4).style("fill", MC(mm));
    });
    // end labels for EchoAlign and EchoPrime only
    ["EchoAlign", "EchoPrime"].forEach((mm, k) => svg.append("text").attr("x", x(100) + 8).attr("y", y(D[mm][5]) + (k ? 10 : -2)).style("font", `${k ? 400 : 600} 11.5px var(--sans)`).style("fill", "var(--ink)").text(mm));
    // crosshair
    const ch = svg.append("line").attr("y1", m.t).attr("y2", H - m.b).style("stroke", "var(--ink-3)").style("opacity", 0);
    svg.append("rect").attr("x", m.l).attr("y", m.t).attr("width", W - m.l - m.r).attr("height", H - m.t - m.b).style("fill", "transparent")
      .on("mousemove", ev => {
        const px = d3.pointer(ev)[0]; const i = d3.minIndex(F, f => Math.abs(x(f) - px));
        ch.attr("x1", x(F[i])).attr("x2", x(F[i])).style("opacity", 1);
        showTip(`<div class="tt">${F[i]}% of training labels</div>${modelTable(MODELS.map(mm => [mm, D[mm][i]]).sort((a, b) => b[1] - a[1]))}`, ev);
      }).on("mouseleave", () => { ch.style("opacity", 0); hideTip(); });
  }
  onVisible(el, () => draw(true)); onResize(() => draw(false));
})();

(function subgroups() {
  const el = $("#subgroups"); let src = "private";
  segmented("sgSrc", v => { src = v; draw(); });
  function draw() {
    el.innerHTML = "";
    const rows = [];
    E.subgroups[src].forEach(([title, items]) => { rows.push({ header: true, label: title }); items.forEach(([n, cnt, ea, ep]) => rows.push({ label: n, cnt, ea, ep })); });
    const W = width(el), rh = 23, m = { l: 150, r: 70, t: 4, b: 28 }, H = m.t + rows.length * rh + m.b;
    const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${W} ${H}`);
    const x = d3.scaleLinear().domain(src === "private" ? [0.80, 0.94] : [0.80, 0.90]).range([m.l, W - m.r]);
    const nt = W < 560 ? 3 : 6;
    svg.append("g").attr("class", "grid").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(nt).tickSize(-(H - m.t - m.b)).tickFormat(""));
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(nt).tickSize(0).tickFormat(d3.format(".2f"))).call(g => g.select(".domain").remove()).selectAll("text").attr("dy", "1em");
    const g = svg.selectAll("g.r").data(rows).join("g").attr("transform", (d, i) => `translate(0,${m.t + i * rh + rh / 2})`);
    g.filter(d => d.header).append("text").attr("x", 2).attr("dy", "0.35em").style("font", "600 12px var(--sans)").style("fill", "var(--ink)").text(d => d.label);
    const r = g.filter(d => !d.header);
    r.append("text").attr("x", 14).attr("dy", "0.35em").style("font", "12px var(--sans)").style("fill", "var(--ink-2)").text(d => d.label);
    r.append("text").attr("x", m.l - 10).attr("dy", "0.35em").attr("text-anchor", "end").style("font", "11px var(--sans)").style("fill", "var(--ink-3)").style("font-variant-numeric", "tabular-nums").text(d => d3.format(",")(d.cnt));
    r.append("line").attr("x1", d => x(d.ep)).attr("x2", d => x(d.ep)).style("stroke", "var(--rule)").style("stroke-width", 3).style("stroke-linecap", "round")
      .transition().duration(DUR).attr("x2", d => x(d.ea));
    r.append("circle").attr("cx", d => x(d.ep)).attr("r", 4.5).style("fill", "var(--m1)");
    r.append("circle").attr("cx", d => x(d.ep)).attr("r", 6).style("fill", "var(--m0)").style("stroke", "var(--card)").style("stroke-width", 1.5)
      .transition().duration(DUR).attr("cx", d => x(d.ea));
    r.append("text").attr("x", d => x(Math.max(d.ea, d.ep)) + 10).attr("dy", "0.35em").style("font", "600 11px var(--sans)").style("font-variant-numeric", "tabular-nums")
      .style("fill", d => d.ea >= d.ep ? "var(--ink)" : "var(--ink-3)").text(d => (d.ea - d.ep >= 0 ? "+" : "") + (d.ea - d.ep).toFixed(3));
    r.append("rect").attr("x", 0).attr("y", -rh / 2).attr("width", W).attr("height", rh).style("fill", "transparent")
      .on("mousemove", (ev, d) => showTip(`<div class="tt">${d.label} · ${d3.format(",")(d.cnt)} studies</div>${modelTable([["EchoAlign", d.ea], ["EchoPrime", d.ep]])}`, ev)).on("mouseleave", hideTip);
  }
  onVisible(el, draw); onResize(draw);
})();

/* =========================================================
   GROUNDING HEATMAP
   ========================================================= */
(function tokens() {
  const el = $("#tokmap"); let src = "private";
  segmented("tkSrc", v => { src = v; draw(); });
  const T = E.tokens, cols = T.cols; // LV RV LA RA AV MV TV PV Peri AoR Arch PA PCA
  function draw() {
    el.innerHTML = "";
    const rows = T.rows.filter(r => r[src][0] != null);
    const W = Math.max(640, width(el)), cw = Math.min(isBig(el) ? 64 : 46, (W - 250) / 13), ch = isBig(el) ? 21 : 17, m = { l: 236, t: 30, b: 10 };
    const H = m.t + (rows.length + 1.4) * ch + m.b;
    const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${m.l + cw * 13 + 20} ${H}`).style("max-width", (m.l + cw * 13 + 20) + "px").style("min-width", (m.l + cw * 13 + 20) + "px");
    const neg = d3.color(cssVar("--div-neg")), mid = d3.color(cssVar("--div-mid")), pos = d3.color(cssVar("--div-pos"));
    const color = v => v == null ? "transparent" : (v < 0 ? d3.interpolateRgb(mid, neg)(Math.min(1, -v / 0.06)) : d3.interpolateRgb(mid, pos)(Math.min(1, v / 0.06)));
    cols.forEach((c, j) => {
      svg.append("text").attr("x", m.l + j * cw + cw / 2 + (j === 12 ? 6 : 0)).attr("y", m.t - 10).attr("text-anchor", "middle").style("font", "600 11px var(--sans)").style("fill", c === "PCA" ? "var(--ink)" : SC(c)).text(c);
    });
    const all = [...rows.map(r => ({ ...r, vals: r[src] })), { name: "Mean of findings", vals: T.mean[src], mean: true }];
    all.forEach((r, i) => {
      const yy = m.t + i * ch + (r.mean ? ch * 0.4 : 0);
      const g = svg.append("g").attr("transform", `translate(0,${yy})`);
      const nm = r.name.replace("Left ventricle", "LV").replace("Right ventricle", "RV");
      g.append("text").attr("x", m.l - 8).attr("y", ch / 2 + 4).attr("text-anchor", "end").style("font", `${r.mean ? 600 : 400} 11.5px var(--sans)`).style("fill", r.mean ? "var(--ink)" : "var(--ink-2)").text(nm.length > 36 ? nm.slice(0, 35) + "…" : nm).append("title").text(r.name);
      const best = r.best ? cols.indexOf(r.best[src]) : -1;
      r.vals.forEach((v, j) => {
        const xx = m.l + j * cw + (j === 12 ? 6 : 0);
        const own = r.structure && cols[j] === r.structure;
        g.append("rect").attr("x", xx + 1).attr("y", 1).attr("width", cw - 2).attr("height", ch - 2).attr("rx", 2)
          .style("fill", color(v)).style("opacity", 0).style("stroke", own ? "var(--ink)" : "none").style("stroke-width", own ? 1.8 : 0)
          .on("mousemove", ev => { groundHover(cols[j]); showTip(`<div class="tt">${r.name}</div>${cols[j] === "PCA" ? "Combined 832-d embedding" : SNAME[cols[j]] + " embedding"}: <b>${v >= 0 ? "+" : ""}${v.toFixed(3)}</b> AUROC vs mean of 12${own ? "<br><i>finding's own structure</i>" : ""}`, ev); }).on("mouseleave", () => { hideTip(); })
          .transition().delay(reduceMotion ? 0 : i * 18 + j * 10).duration(400).style("opacity", 1);
        if (!r.mean && j === best) g.append("circle").attr("cx", xx + cw / 2).attr("cy", ch / 2).attr("r", 2.2).style("fill", "var(--ink)").style("pointer-events", "none");
      });
    });
    const own = rows.filter(r => r.structure);
    const ownBeat = own.filter(r => r[src][cols.indexOf(r.structure)] > 0).length;
    const ownBest = own.filter(r => r.best[src] === r.structure).length;
    $("#tkLegend").innerHTML = `<div class="legend"><span class="k"><span class="sw" style="border-radius:2px;background:var(--div-neg)"></span>−0.06</span><span class="k"><span class="sw" style="border-radius:2px;background:var(--div-mid);border:1px solid var(--rule)"></span>0</span><span class="k"><span class="sw" style="border-radius:2px;background:var(--div-pos)"></span>+0.06</span><span class="k"><span class="sw" style="border-radius:2px;border:2px solid var(--ink)"></span>own structure</span><span class="k">● best of 12</span></div>
    <div class="small muted" style="margin-top:4px">Own structure above the mean on <b>${ownBeat} of ${own.length}</b> findings · highest of the 12 on <b>${ownBest}</b></div>`;
  }
  onVisible(el, draw); onResize(draw); themeHooks.push(() => setTimeout(draw, 30));
  matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", () => setTimeout(draw, 30));
})();

/* =========================================================
   DEMO: Hugging Face Space
   ========================================================= */
(function demo() {
  const params = new URLSearchParams(location.search);
  const SPACE = params.get("space") || window.ECHOALIGN_SPACE || "EchoAlign/demo";
  const body = $("#demoBody"), url = $("#demoUrl"), open = $("#demoOpen");
  const host = `https://${SPACE.toLowerCase().replace("/", "-").replace(/[._]/g, "-")}.hf.space`;
  const page = `https://huggingface.co/spaces/${SPACE}`;
  url.textContent = innerWidth < 520 ? SPACE : `huggingface.co/spaces/${SPACE}`; open.href = page;
  const isDark = () => document.documentElement.dataset.theme === "dark" || (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
  let mounted = false;
  const beating = (msg) => `<div class="demo-placeholder demo-loading"><div class="pump-logo"></div><div class="note">${msg}</div></div>`;
  const startLoaders = () => body.querySelectorAll(".pump-logo:not(.on)").forEach(el => { el.classList.add("on"); alignLoader(el); });
  function embed() {
    if (mounted) return; mounted = true;
    body.innerHTML = beating(`Loading the demo… the Space may take a minute to wake up.`) +
      `<iframe src="${host}/?__theme=${isDark() ? "dark" : "light"}" title="EchoAlign demo" allow="clipboard-read; clipboard-write; fullscreen; camera; microphone" class="loading"></iframe>`;
    startLoaders();
    const f = body.querySelector("iframe");
    f.addEventListener("load", () => { f.classList.remove("loading"); body.querySelector(".demo-loading")?.remove(); }, { once: true });
  }
  themeHooks.push(() => { const f = body.querySelector("iframe"); if (f) f.src = `${host}/?__theme=${isDark() ? "dark" : "light"}`; });
  function notice() {
    body.innerHTML = `<div class="demo-placeholder">
      <div class="pump-logo"></div>
      <div class="ct">The demo is not public yet</div>
      <p>The Space <span class="mono">${SPACE}</span> embeds here automatically once it is public. Members of the EchoAlign organization can open it on Hugging Face.</p>
      <div class="btnrow" style="justify-content:center;margin:0">
        <a class="btn primary" href="${page}" target="_blank" rel="noopener">Open the Space ↗</a>
        <button class="btn" id="demoForce">Try embedding anyway</button>
      </div></div>`;
    startLoaders();
    $("#demoForce").addEventListener("click", embed);
  }
  body.innerHTML = beating(`Connecting to ${SPACE}…`); startLoaders();
  (async () => {
    let ok = false;
    try { const r = await fetch(`https://huggingface.co/api/spaces/${SPACE}`); ok = r.ok; } catch (e) { ok = false; }
    if (ok) onVisible(body, embed, "300px"); else notice();
  })();
})();

})();
