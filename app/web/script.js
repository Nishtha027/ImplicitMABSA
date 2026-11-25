/* ImplicitMABSA — web UI (shared by index.html and lab.html)
 * The lab calls the local Python server at POST /api/predict (FormData: dataset, tweet, aspect, image).
 * The mock is off: to try the lab without the server, open lab.html?mock and it returns the sample response.
 */
const USE_MOCK = new URLSearchParams(location.search).has("mock");

const MOCK_RESPONSE = {
  sentiment: { negative: 0.05, neutral: 0.15, positive: 0.80 },
  aspect_type: { person: 0.70, location: 0.10, organisation: 0.15, other: 0.05 },
  text_seen: "Just got my new <mask> phone, so happy!",
  aspect_hidden: true,
  models: {
    sentiment: { model: "concat", test_acc: 0.773, macro_f1: 0.728 },
    aspect_type: { model: "concat", test_acc: 0.727, macro_f1: 0.655 },
  },
};

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const root = document.documentElement;
const PAGE = document.body.dataset.page;
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const pct = (v) => `${Math.round(v * 100)}%`;
const ss = { get: (k) => { try { return sessionStorage.getItem(k); } catch (e) { return null; } }, set: (k, v) => { try { sessionStorage.setItem(k, v); } catch (e) { /* blocked */ } }, del: (k) => { try { sessionStorage.removeItem(k); } catch (e) { /* blocked */ } } };

function replay(el, cls) {
  if (!el || reduceMotion) return;
  el.classList.remove(cls); void el.offsetWidth; el.classList.add(cls);
  el.addEventListener("animationend", () => el.classList.remove(cls), { once: true });
}
function countUp(el, v, fmt, dur = 1000) {
  if (reduceMotion) { el.textContent = fmt(v); return; }
  const t0 = performance.now();
  (function f(now) {
    const t = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - t, 3);
    el.textContent = fmt(v * e);
    if (t < 1) requestAnimationFrame(f);
  })(t0);
}

/* ------------------------------------------------------------------ */
/* Pixel art, 14 x 14: one character per ViT patch                     */
/* ------------------------------------------------------------------ */
const PALETTE = {
  B: "#E9A23B", b: "#C9801E", s: "#FFF2D0", L: "#3FA34D", C: "#F7C948", M: "#6B3A26", T: "#E5484D",
  S: "#BFD8FF", W: "#FFFFFF", K: "#20233A", G: "#4CAF50", g: "#3A9142", E: "#6B4E3D", Y: "#FFD166",
};
const EXAMPLES = [
  { tweet: "Dinner at Joe's Diner tonight. Absolutely delicious!", aspect: "Joe's Diner", dataset: "twitter2015", file: "joes-diner.png",
    art: ["..............", ".....BBBB.....", "...BBBsBBBB...", "..BBsBBBBBsB..", ".BBBBBBsBBBBB.", ".BBsBBBBBBsBB.", ".bbbbbbbbbbbb.", "LLLLLLLLLLLLLL", ".CCCCCCCCCCCC.", ".MMMMMMMMMMMM.", ".MMMMMMMMMMMM.", ".TTTTTTTTTTTT.", ".bbbbbbbbbbbb.", "..BBBBBBBBBB.."] },
  { tweet: "What a performance from Arsenal tonight, the fans were unreal!", aspect: "Arsenal", dataset: "twitter2017", file: "arsenal.png",
    art: ["SSSSSSSSSSSSSS", "SSSSSWWWWSSSSS", "SSSWWWKKWWWSSS", "SSWWWKKKKWWWSS", "SSWKWWKKWWKWSS", "SWKKWWWWWWKKWS", "SWKWWWWWWWWKWS", "SWWWWKKKKWWWWS", "SSWWKKKKKKWWSS", "SSWWWKKKKWWWSS", "SSSWWWWWWWWSSS", "GGGGGWWWWGGGGG", "GgGGGGGGGGgGGG", "GGGgGGGGGgGGGG"] },
  { tweet: "Sunset views from Paris never get old.", aspect: "Paris", dataset: "twitter2015", file: "paris.png",
    art: ["SSSSSSESSSSSYY", "SSSSSSESSSSSYY", "SSSSSEEESSSSSS", "SSSSSESESSSSSS", "SSSSEEEEESSSSS", "SSSSSESESSSSSS", "SSSSESSSESSSSS", "SSSEEEEEEESSSS", "SSSESSSSSESSSS", "SSESSSSSSSESSS", "SSESSEEESSESSS", "SESSESSSESSESS", "GEGGEGGGEGGEGG", "GGGGGGGGGGGGGG"] },
];
function drawArt(art, cell, voidColor) {
  const cv = document.createElement("canvas");
  cv.width = cv.height = 14 * cell;
  const ctx = cv.getContext("2d");
  art.forEach((row, r) => [...row].forEach((ch, c) => { ctx.fillStyle = PALETTE[ch] || voidColor; ctx.fillRect(c * cell, r * cell, cell, cell); }));
  return cv;
}

/* ------------------------------------------------------------------ */
/* Page curtain: the screen dissolves into 196 patches between pages    */
/* ------------------------------------------------------------------ */
const GRID = 14;
const curtain = (() => {
  const el = $("#curtain"), name = $("#curtainName"), sub = $("#curtainSub");
  const tiles = [];
  for (let i = 0; i < GRID * GRID; i++) { const t = document.createElement("i"); el.appendChild(t); tiles.push(t); }
  const setDelays = (ox, oy, spread) => tiles.forEach((t, i) => {
    const r = Math.floor(i / GRID) / (GRID - 1), c = (i % GRID) / (GRID - 1);
    t.style.setProperty("--d", `${(Math.hypot(c - ox, r - oy) * spread + Math.random() * 0.08).toFixed(3)}s`);
  });
  let onRevealed = [];
  const whenRevealed = (fn) => (onRevealed ? onRevealed.push(fn) : fn());
  const done = () => { const fns = onRevealed; onRevealed = null; fns.forEach((f) => f()); };

  async function reveal() {
    if (!root.classList.contains("arriving")) { done(); return; }
    const firstVisit = !ss.get("imabsa-seen");
    ss.set("imabsa-seen", "1");
    ss.del("imabsa-nav");
    if (firstVisit) {
      // short preloader: count the patches in
      name.textContent = "ImplicitMABSA";
      const t0 = performance.now(), dur = 1100;
      await new Promise((res) => (function f(now) {
        const t = Math.min(1, (now - t0) / dur);
        sub.textContent = `${Math.round(196 * (1 - Math.pow(1 - t, 2)))} / 196 patches`;
        t < 1 ? requestAnimationFrame(f) : res();
      })(t0));
      await sleep(150);
    } else {
      await sleep(260);
    }
    setDelays(0.5, 0.5, 0.55);
    el.classList.add("cover");
    root.classList.remove("arriving");
    // arriving on index.html#how: jump to the section while the patches still cover the screen
    const target = location.hash && document.getElementById(location.hash.slice(1));
    if (target) window.scrollTo({ top: scrollY + target.getBoundingClientRect().top, behavior: "instant" });
    el.style.background = "transparent";
    void el.offsetWidth;
    el.classList.remove("cover");
    setTimeout(done, 350);
    setTimeout(() => (el.style.background = ""), 1200);
  }

  function leave(href, label, x, y) {
    if (reduceMotion) { location.href = href; return; }
    name.textContent = label;
    sub.textContent = "reading 196 patches";
    setDelays(x / innerWidth, y / innerHeight, 0.45);
    root.classList.add("covering");
    el.classList.add("cover");
    ss.set("imabsa-nav", "1");
    setTimeout(() => (location.href = href), 820);
  }

  // coming back with the browser's back button restores the page from cache: uncover it
  window.addEventListener("pageshow", (e) => {
    if (e.persisted) { el.classList.remove("cover"); root.classList.remove("covering", "arriving"); ss.del("imabsa-nav"); }
  });

  const LABELS = { "index.html": "Home", "lab.html": "Lab" };
  const scrollToHash = (hash) => {
    const target = hash && document.getElementById(hash.slice(1));
    if (target) target.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
    else scrollTo({ top: 0, behavior: reduceMotion ? "auto" : "smooth" });
  };
  document.addEventListener("click", (e) => {
    const a = e.target.closest("a[href]");
    if (!a || e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || a.target === "_blank") return;
    const url = new URL(a.getAttribute("href"), location.href);
    if (url.origin !== location.origin) return;
    const file = url.pathname.split("/").pop() || "index.html";
    if (!LABELS[file]) return;
    if (USE_MOCK && !url.search) url.search = "?mock"; // keep the mock on while browsing between pages
    const here = location.pathname.split("/").pop() || "index.html";
    e.preventDefault();
    if (file === here) {
      // same page: just glide to the section (or the top)
      if (url.hash) history.replaceState(null, "", url.hash); else history.replaceState(null, "", location.pathname + location.search);
      scrollToHash(url.hash);
      return;
    }
    leave(url.href, LABELS[file], e.clientX || innerWidth / 2, e.clientY || innerHeight / 2);
  });

  return { reveal, whenRevealed };
})();
// lift the curtain right away, so nothing later on the page can ever leave it stuck
curtain.reveal();

/* ------------------------------------------------------------------ */
/* Theme: pull cord lamp + eclipse reveal                              */
/* ------------------------------------------------------------------ */
const isDark = () => root.dataset.theme !== "light";
function setTheme(next, x, y) {
  const apply = () => {
    if (next === "light") root.dataset.theme = "light"; else delete root.dataset.theme;
    try { localStorage.setItem("imabsa-theme", next); } catch (e) { /* storage blocked */ }
    $("#cord").setAttribute("aria-label", next === "dark" ? "Switch to light theme" : "Switch to dark theme");
  };
  if (!document.startViewTransition || reduceMotion) { apply(); return; }
  const r = Math.hypot(Math.max(x, innerWidth - x), Math.max(y, innerHeight - y));
  const vt = document.startViewTransition(apply);
  vt.ready.then(() => root.animate(
    { clipPath: [`circle(0px at ${x}px ${y}px)`, `circle(${r}px at ${x}px ${y}px)`] },
    { duration: 750, easing: "cubic-bezier(.2,.8,.2,1)", pseudoElement: "::view-transition-new(root)" },
  ));
}
const toggleTheme = (x, y) => setTheme(isDark() ? "light" : "dark", x, y);

(function cord() {
  const btn = $("#cord"), path = $("#cordPath"), knob = $("#cordKnob");
  if (!btn) return;
  btn.setAttribute("aria-label", isDark() ? "Switch to light theme" : "Switch to dark theme");
  const REST = 70;
  let len = REST, sx = 0, vy = 0, vx = 0, raf = 0, drag = null, justDragged = false;
  const render = () => {
    path.setAttribute("d", `M20 0 Q ${20 + sx * 0.4} ${len * 0.55} ${20 + sx} ${len}`);
    knob.style.transform = `translate(${sx}px, ${len - 6}px)`;
  };
  function spring() {
    vy += (REST - len) * 0.18; vy *= 0.78; vx += -sx * 0.12; vx *= 0.86;
    len += vy; sx += vx; render();
    if (Math.abs(vy) + Math.abs(len - REST) + Math.abs(vx) + Math.abs(sx) > 0.15) raf = requestAnimationFrame(spring);
    else { len = REST; sx = 0; render(); raf = 0; }
  }
  const release = () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(spring); };
  const knobPoint = () => { const r = knob.getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2]; };
  btn.addEventListener("pointerdown", (e) => { cancelAnimationFrame(raf); drag = { y0: e.clientY, x0: e.clientX, moved: 0 }; btn.setPointerCapture(e.pointerId); });
  btn.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const dy = e.clientY - drag.y0, dx = e.clientX - drag.x0;
    drag.moved = Math.max(drag.moved, Math.hypot(dx, dy));
    len = REST + clamp(dy, -10, 90) * 0.85; sx = clamp(dx, -40, 40) * 0.5; render();
  });
  btn.addEventListener("pointerup", () => {
    if (!drag) return;
    const pulled = len - REST, moved = drag.moved;
    drag = null;
    if (moved > 4) justDragged = true;
    if (pulled > 28) { const [x, y] = knobPoint(); toggleTheme(x, y); }
    vy = pulled > 28 ? -6 : 0; release();
  });
  btn.addEventListener("click", () => {
    if (justDragged) { justDragged = false; return; }
    len = REST + 38; render();
    const [x, y] = knobPoint(); toggleTheme(x, y);
    vy = -4; release();
  });
  render();
})();

/* ------------------------------------------------------------------ */
/* Header: hides on scroll down, gooey pill on the current page, menu  */
/* ------------------------------------------------------------------ */
(function header() {
  const bar = $(".bar");
  let lastY = scrollY;
  const onScroll = () => {
    const y = scrollY;
    bar.classList.toggle("scrolled", y > 8);
    bar.classList.toggle("hide", y > 260 && y > lastY + 4 && $("#mobileMenu").hidden);
    if (y < lastY - 4 || y < 260) bar.classList.remove("hide");
    lastY = y;
  };
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  const nav = $("#nav");
  const links = $$("a[data-page]", nav);
  const blobs = $$(".blob", nav);
  let current = links.find((l) => l.dataset.page === PAGE) || null;
  let hovering = false;
  function moveTo(link) {
    links.forEach((l) => l.classList.toggle("lit", l === link));
    blobs.forEach((b) => {
      if (!link) { b.style.setProperty("--o", 0); return; }
      b.style.setProperty("--x", `${link.offsetLeft}px`);
      b.style.setProperty("--w", `${link.offsetWidth}px`);
      b.style.setProperty("--o", 1);
    });
  }
  links.forEach((l) => l.addEventListener("pointerenter", () => { hovering = true; moveTo(l); }));
  nav.addEventListener("pointerleave", () => { hovering = false; moveTo(current); });
  if (current) current.setAttribute("aria-current", "page");
  // on the home page the pill follows the reader into "How it works" and back
  if (PAGE === "home") {
    const homeLink = links.find((l) => l.dataset.page === "home"), howLink = links.find((l) => l.dataset.page === "how");
    new IntersectionObserver((en) => {
      current = en[0].isIntersecting ? howLink : homeLink;
      if (!hovering) moveTo(current);
    }, { rootMargin: "-45% 0px -50% 0px" }).observe($("#how"));
  }
  const settle = () => moveTo(current);
  settle();
  document.fonts && document.fonts.ready.then(settle);
  window.addEventListener("resize", settle);

  const menuBtn = $("#menuBtn"), menu = $("#mobileMenu");
  $$("a[data-page]", menu).forEach((a) => a.classList.toggle("lit", a.dataset.page === PAGE));
  const setMenu = (open) => { menu.hidden = !open; menuBtn.setAttribute("aria-expanded", String(open)); menuBtn.setAttribute("aria-label", open ? "Close menu" : "Open menu"); };
  menuBtn.addEventListener("click", () => setMenu(menu.hidden));
  menu.addEventListener("click", (e) => { if (e.target.closest("a")) setMenu(false); });
  $("#mmTheme").addEventListener("click", () => { const r = menuBtn.getBoundingClientRect(); toggleTheme(r.left + r.width / 2, r.top + r.height / 2); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !menu.hidden) { setMenu(false); menuBtn.focus(); } });
})();

/* ------------------------------------------------------------------ */
/* Magnetic buttons                                                    */
/* ------------------------------------------------------------------ */
(function magnetic() {
  if (reduceMotion || !matchMedia("(hover: hover)").matches) return;
  $$("[data-magnetic]").forEach((el) => {
    const inner = el.firstElementChild;
    el.addEventListener("pointermove", (e) => {
      const r = el.getBoundingClientRect();
      const dx = e.clientX - (r.left + r.width / 2), dy = e.clientY - (r.top + r.height / 2);
      el.style.transition = "transform .15s linear, box-shadow .3s, background .2s, color .3s";
      el.style.transform = `translate(${dx * 0.28}px, ${dy * 0.38}px)`;
      if (inner) inner.style.transform = `translate(${dx * 0.12}px, ${dy * 0.16}px)`;
    });
    el.addEventListener("pointerleave", () => {
      el.style.transition = "transform .7s var(--spring), box-shadow .3s, background .2s, color .3s";
      el.style.transform = "";
      if (inner) inner.style.transform = "";
    });
    el.addEventListener("pointerdown", () => replay(inner || el, "jelly"));
  });
})();

/* ------------------------------------------------------------------ */
/* Scroll reveals (titles slide up line by line), started after the curtain */
/* ------------------------------------------------------------------ */
curtain.whenRevealed(() => {
  const els = $$(".reveal, .rise");
  if (reduceMotion || !("IntersectionObserver" in window)) { els.forEach((el) => el.classList.add("in")); return; }
  const io = new IntersectionObserver((entries) => entries.forEach((en) => {
    if (en.isIntersecting) { en.target.classList.add("in"); io.unobserve(en.target); }
  }), { threshold: 0.12, rootMargin: "0px 0px -6% 0px" });
  els.forEach((el, i) => { if (el.classList.contains("rise")) el.style.transitionDelay = el.style.transitionDelay || `${(i % 4) * 0.07}s`; io.observe(el); });
});

/* CTA band: patches light up under the pointer */
(function ctaGrid() {
  const grid = $("#ctaGrid");
  if (!grid) return;
  const cta = grid.parentElement;
  let cells = [];
  const build = () => {
    const cols = 28, size = cta.clientWidth / cols, rows = Math.ceil(cta.clientHeight / size);
    grid.innerHTML = "";
    cells = Array.from({ length: cols * rows }, () => grid.appendChild(document.createElement("i")));
  };
  build();
  window.addEventListener("resize", build);
  if (reduceMotion) return;
  cta.addEventListener("pointermove", (e) => {
    const r = grid.getBoundingClientRect(), cols = 28, size = r.width / cols;
    const c = Math.floor((e.clientX - r.left) / size), rw = Math.floor((e.clientY - r.top) / size);
    const cell = cells[rw * cols + c];
    if (!cell) return;
    cell.classList.add("on");
    setTimeout(() => cell.classList.remove("on"), 250);
  });
})();

/* ------------------------------------------------------------------ */
/* Patch grids (shared by home demo, method story and the lab)         */
/* ------------------------------------------------------------------ */
function buildPatches(container, paint) {
  container.innerHTML = "";
  const frag = document.createDocumentFragment();
  const c = (GRID - 1) / 2;
  for (let r = 0; r < GRID; r++) for (let col = 0; col < GRID; col++) {
    const p = document.createElement("div");
    p.className = "patch";
    const ang = Math.random() * Math.PI * 2, dist = 400 + Math.random() * 900;
    p.style.setProperty("--sx", `${Math.cos(ang) * dist}%`);
    p.style.setProperty("--sy", `${Math.sin(ang) * dist}%`);
    p.style.setProperty("--sr", `${(Math.random() - 0.5) * 220}deg`);
    p.style.setProperty("--d", `${(Math.hypot(r - c, col - c) * 0.045 + Math.random() * 0.12).toFixed(3)}s`);
    paint(p, r, col);
    frag.appendChild(p);
  }
  container.appendChild(frag);
  return [...container.children];
}
function paintArt(art) {
  return (p, r, c) => { const col = PALETTE[art[r][c]]; if (col) p.style.background = col; else p.classList.add("patch-void"); };
}
async function assemble(container) {
  container.classList.remove("gridded");
  container.classList.add("scatter");
  if (reduceMotion) { container.classList.remove("scatter"); return; }
  void container.offsetWidth;
  await sleep(30);
  container.classList.remove("scatter");
  await sleep(1500);
}
function attentionFlicker(patches, ms) {
  if (reduceMotion || !patches.length) return Promise.resolve();
  return new Promise((resolve) => {
    const end = performance.now() + ms, lit = new Set();
    (function step() {
      lit.forEach((p) => p.classList.remove("hot")); lit.clear();
      if (performance.now() > end) return resolve();
      const n = 6 + ((Math.random() * 8) | 0);
      for (let k = 0; k < n; k++) { const p = patches[(Math.random() * patches.length) | 0]; p.classList.add("hot"); lit.add(p); }
      setTimeout(step, 140);
    })();
  });
}

/* Segmented control with a stretchy thumb (lab dataset, research dataset) */
function initSeg(seg, onChange) {
  const inputs = $$("input", seg);
  const sync = (dir) => {
    const idx = inputs.findIndex((i) => i.checked);
    seg.classList.toggle("go-right", dir === "right");
    seg.classList.toggle("go-left", dir === "left");
    seg.classList.toggle("at-2", idx === 1);
  };
  seg.addEventListener("change", (e) => { sync(inputs.indexOf(e.target) === 1 ? "right" : "left"); onChange && onChange(e.target.value); });
  sync();
}
/* ================================================================== */
/* HOME                                                                */
/* ================================================================== */
if (PAGE === "home") {
  // hide-a-word playground in the call-to-action card (an illustration, not the trained model)
  (function playground() {
    const tweetEl = $("#playTweet");
    if (!tweetEl) return;
    const WORDS = [
      { t: "Absolutely", w: 1 }, { t: "loved", w: 3 }, { t: "dinner", w: 0 }, { t: "at", w: 0 },
      { t: "Joe's Diner", w: 0, start: true }, { t: "tonight!", w: 0 },
    ];
    const rows = $$(".pm-row", $("#playMeter"));
    const verdict = $("#playVerdict");
    const buttons = WORDS.map((wd) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "word" + (wd.w ? " opinion" : "");
      tweetEl.appendChild(b);
      return b;
    });
    let masked = new Set();
    function render() {
      buttons.forEach((b, i) => {
        const on = masked.has(i);
        b.classList.toggle("masked", on);
        b.textContent = on ? "<mask>" : WORDS[i].t;
        b.setAttribute("aria-pressed", String(on));
        b.setAttribute("aria-label", on ? `Show "${WORDS[i].t}"` : `Hide "${WORDS[i].t}"`);
      });
      const score = WORDS.reduce((s, wd, i) => s + (masked.has(i) ? 0 : wd.w), 0);
      const pos = 0.12 + 0.185 * score, neg = 0.03, neu = 1 - pos - neg;
      const probs = { positive: pos, neutral: neu, negative: neg };
      const top = pos >= neu ? "positive" : "neutral";
      rows.forEach((r) => {
        const v = probs[r.dataset.k];
        $("b", r).style.width = `${(v * 100).toFixed(1)}%`;
        $("em", r).textContent = pct(v);
        r.classList.toggle("top", r.dataset.k === top);
      });
      verdict.innerHTML = top === "positive"
        ? "The words left still say <strong>positive</strong>"
        : "With the opinion hidden, it reads <strong>neutral</strong>";
    }
    const reset = () => { masked = new Set(WORDS.map((w, i) => (w.start ? i : -1)).filter((i) => i >= 0)); render(); };
    buttons.forEach((b, i) => b.addEventListener("click", () => { if (masked.has(i)) masked.delete(i); else masked.add(i); render(); }));
    $("#playReset").addEventListener("click", reset);
    reset();
  })();

  (function heroMask() {
    const el = $("#heroMask");
    const words = ["word", "brand", "place", "team", "person"];
    if (reduceMotion) { el.textContent = "<mask>"; el.classList.add("is-masked"); return; }
    let i = 0;
    const scramble = async (target) => {
      const glyphs = "abcdefghijklmnopqrstuvwxyz";
      for (let f = 0; f < 8; f++) { el.textContent = target.split("").map((c, k) => (k < f - 2 ? c : glyphs[(Math.random() * 26) | 0])).join(""); await sleep(45); }
      el.textContent = target;
    };
    curtain.whenRevealed(async () => {
      await sleep(1600);
      while (true) {
        el.classList.add("is-masked"); el.textContent = "<mask>";
        await sleep(1500);
        i = (i + 1) % words.length;
        el.classList.remove("is-masked");
        await scramble(words[i]);
        await sleep(2000);
      }
    });
  })();

  (function demo() {
    const tweetEl = $("#demoTweet"), patchesEl = $("#demoPatches"), photo = $(".demo-photo"), out = $("#demoOut"), card = $(".demo-card");
    const patches = buildPatches(patchesEl, paintArt(EXAMPLES[0].art));
    const before = "Dinner at ", aspect = "Joe's Diner", after = " tonight. Absolutely delicious!";
    const bars = $$(".demo-row b", out);
    if (reduceMotion) {
      tweetEl.innerHTML = `${escapeHtml(before)}<span class="mask-chip">&lt;mask&gt;</span>${escapeHtml(after)}`;
      bars.forEach((b) => (b.style.width = `${b.dataset.v * 100}%`));
      out.classList.add("show"); patchesEl.classList.add("gridded");
      return;
    }
    patchesEl.classList.add("scatter");
    const type = async (text, el) => { for (const ch of text) { el.insertAdjacentText("beforeend", ch); await sleep(ch === " " ? 40 : 28 + Math.random() * 40); } };
    curtain.whenRevealed(async () => {
      await sleep(500);
      while (true) {
        out.classList.remove("show");
        bars.forEach((b) => (b.style.width = "0"));
        tweetEl.innerHTML = "";
        patchesEl.classList.remove("gridded"); patchesEl.classList.add("scatter");
        const textNode = document.createElement("span"), caret = document.createElement("span");
        caret.className = "caret"; tweetEl.append(textNode, caret);
        await sleep(400);
        await type(before, textNode);
        const asp = document.createElement("span"); asp.className = "aspect-span"; textNode.appendChild(asp);
        await type(aspect, asp);
        const rest = document.createElement("span"); textNode.appendChild(rest);
        await type(after, rest);
        caret.remove();
        const assembled = assemble(patchesEl);
        await sleep(500); asp.classList.add("is-target");
        await sleep(900); asp.outerHTML = '<span class="mask-chip">&lt;mask&gt;</span>';
        await assembled;
        patchesEl.classList.add("gridded"); photo.classList.add("scanning");
        await attentionFlicker(patches, 1800);
        photo.classList.remove("scanning");
        bars.forEach((b, i) => setTimeout(() => (b.style.width = `${b.dataset.v * 100}%`), i * 120));
        out.classList.add("show");
        await sleep(4200);
      }
    });
    const fig = $(".demo");
    fig.addEventListener("pointermove", (e) => {
      const r = fig.getBoundingClientRect(), x = (e.clientX - r.left) / r.width - 0.5, y = (e.clientY - r.top) / r.height - 0.5;
      card.style.transform = `rotateY(${x * 10}deg) rotateX(${-y * 10}deg)`;
    });
    fig.addEventListener("pointerleave", () => (card.style.transform = ""));
  })();

  (function story() {
    const stage = $("#stage");
    const tokens = $("#tokens");
    ["Dinner", "at", "<mask>", "tonight", ".", "Abs", "olutely", "delicious", "!"].forEach((t, i) => {
      const el = document.createElement("i");
      el.style.setProperty("--h", `${28 + ((i * 37) % 60)}px`);
      el.style.setProperty("--d", `${i * 0.05}s`);
      if (t === "<mask>") el.className = "m";
      el.title = t;
      tokens.appendChild(el);
    });
    const sp = $("#storyPatches");
    buildPatches(sp, paintArt(EXAMPLES[0].art));
    sp.classList.add("scatter");
    let assembled = false;
    const blocks = $$(".st-block", stage);
    function setStep(n) {
      stage.dataset.step = n;
      blocks.forEach((b) => { const k = +b.dataset.b; b.classList.toggle("on", k === n); b.classList.toggle("done", k < n); });
      if (n >= 3 && !assembled) { assembled = true; assemble(sp).then(() => sp.classList.add("gridded")); }
      if (n < 3 && assembled) { assembled = false; sp.classList.remove("gridded"); sp.classList.add("scatter"); }
    }
    const steps = $$(".story-step");
    const io = new IntersectionObserver((entries) => entries.forEach((en) => {
      if (en.isIntersecting) { steps.forEach((s) => s.classList.toggle("on", s === en.target)); setStep(+en.target.dataset.step); }
    }), { rootMargin: "-45% 0px -45% 0px" });
    steps.forEach((s) => io.observe(s));
    const rail = $$(".stage-rail span", stage);
    const onScroll = () => {
      steps.forEach((s, i) => {
        const r = s.getBoundingClientRect();
        rail[i].style.setProperty("--p", clamp((innerHeight / 2 - r.top) / r.height, 0, 1).toFixed(3));
      });
      if (steps[0].getBoundingClientRect().top > innerHeight / 2) setStep(0);
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
  })();
}

/* ================================================================== */
/* LAB                                                                 */
/* ================================================================== */
if (PAGE === "lab") {
  const form = $("#form"), tweet = $("#tweet"), aspectIn = $("#aspect"), mirror = $("#tweetMirror");
  const maskHint = $("#maskHint"), count = $("#count"), fileIn = $("#image"), drop = $("#drop");
  const dropEmpty = $("#dropEmpty"), dropFull = $("#dropFull"), livePatches = $("#livePatches");
  const fileName = $("#fileName"), clearImg = $("#clearImg"), formError = $("#formError"), seg = $("#seg");
  let imageFile = null, livePatchEls = [];

  initSeg(seg);
  const setDataset = (v) => {
    const input = form.querySelector(`input[name="dataset"][value="${v}"]`);
    if (input && !input.checked) { input.checked = true; input.dispatchEvent(new Event("change", { bubbles: true })); }
  };

  function updateMirror() {
    const text = tweet.value, a = aspectIn.value.trim();
    count.textContent = `${text.length} / 400`;
    let found = false, html = escapeHtml(text);
    if (a) html = html.replace(new RegExp(escapeRe(escapeHtml(a)), "gi"), (m) => { found = true; return `<mark>${m}</mark>`; });
    mirror.innerHTML = html + "\n";
    mirror.scrollTop = tweet.scrollTop;
    maskHint.classList.toggle("on", found);
    if (!a) maskHint.textContent = "Name an aspect to hide it from the model.";
    else if (found) maskHint.textContent = `"${a}" is in the tweet, so the model will see <mask> instead.`;
    else maskHint.textContent = `"${a}" isn't in the tweet, so the model sees the tweet as written.`;
  }
  const showFormError = (msg) => { formError.textContent = msg; formError.hidden = false; };
  const hideFormError = () => { formError.hidden = true; };
  tweet.addEventListener("input", () => { updateMirror(); tweet.closest(".field").classList.remove("invalid"); hideFormError(); });
  tweet.addEventListener("scroll", () => (mirror.scrollTop = tweet.scrollTop));
  aspectIn.addEventListener("input", updateMirror);
  updateMirror();

  function loadImage(file) {
    if (!file) return;
    if (!/^image\/(png|jpe?g)$/i.test(file.type)) { showFormError("Choose a JPG or PNG image."); return; }
    hideFormError();
    const url = URL.createObjectURL(file), img = new Image();
    img.onload = () => {
      const S = 448, cv = document.createElement("canvas");
      cv.width = cv.height = S;
      const side = Math.min(img.naturalWidth, img.naturalHeight);
      cv.getContext("2d").drawImage(img, (img.naturalWidth - side) / 2, (img.naturalHeight - side) / 2, side, side, 0, 0, S, S);
      URL.revokeObjectURL(url);
      const data = cv.toDataURL("image/jpeg", 0.9);
      imageFile = file;
      drop.classList.add("has-img");
      drop.closest(".field").classList.remove("invalid");
      dropEmpty.hidden = true; dropFull.hidden = false;
      fileName.textContent = file.name; clearImg.hidden = false;
      livePatchEls = buildPatches(livePatches, (p, r, c) => {
        p.style.backgroundImage = `url(${data})`;
        p.style.backgroundSize = `${GRID * 100}% ${GRID * 100}%`;
        p.style.backgroundPosition = `${(c / (GRID - 1)) * 100}% ${(r / (GRID - 1)) * 100}%`;
      });
      assemble(livePatches).then(() => livePatches.classList.add("gridded"));
    };
    img.onerror = () => { URL.revokeObjectURL(url); showFormError("That file couldn't be read as an image."); };
    img.src = url;
  }
  fileIn.addEventListener("change", () => loadImage(fileIn.files[0]));
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) loadImage(f); });
  clearImg.addEventListener("click", () => {
    imageFile = null; fileIn.value = ""; livePatches.innerHTML = ""; livePatchEls = [];
    livePatches.classList.remove("gridded"); drop.classList.remove("has-img");
    dropEmpty.hidden = false; dropFull.hidden = true; fileName.textContent = ""; clearImg.hidden = true;
  });

  // example card fan
  EXAMPLES.forEach((ex) => {
    const card = document.createElement("button");
    card.type = "button"; card.className = "fan-card"; card.setAttribute("role", "listitem");
    card.setAttribute("aria-label", `Use example: ${ex.tweet}`);
    const txt = document.createElement("span"); txt.className = "fc-text"; txt.textContent = ex.tweet;
    const asp = document.createElement("span"); asp.className = "fc-asp"; asp.innerHTML = `aspect <b>${escapeHtml(ex.aspect)}</b>`;
    card.append(drawArt(ex.art, 1, "#F1F2F7"), txt, asp);
    card.addEventListener("click", () => {
      replay(card, "picked");
      tweet.value = ex.tweet; aspectIn.value = ex.aspect; setDataset(ex.dataset); updateMirror();
      tweet.closest(".field").classList.remove("invalid");
      drawArt(ex.art, 32, "#F1F2F7").toBlob((blob) => loadImage(new File([blob], ex.file, { type: "image/png" })), "image/png");
      const r = form.getBoundingClientRect();
      if (r.top < 0 || r.top > innerHeight * 0.6) form.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
    });
    $("#fan").appendChild(card);
  });

  // swipe to confirm + paper plane
  const swipe = $("#swipe"), knobEl = $("#swipeKnob"), swipeText = $("#swipeText"), plane = $("#plane");
  const swipeMax = () => swipe.clientWidth - 64;
  let swipeBusy = false;
  const setKnob = (px, animate) => {
    swipe.classList.toggle("settle", !!animate);
    const m = swipeMax(), k = clamp(px, 0, m);
    swipe.style.setProperty("--k", `${k}px`); swipe.style.setProperty("--kp", (k / m).toFixed(3));
  };
  const resetSwipe = () => { swipe.classList.remove("done"); swipeText.textContent = "Slide to predict"; plane.classList.remove("fly"); setKnob(0, true); swipeBusy = false; };
  (function swipeDrag() {
    let start = null, dragged = false;
    knobEl.addEventListener("pointerdown", (e) => {
      if (swipeBusy) return;
      start = { x: e.clientX, k: parseFloat(getComputedStyle(swipe).getPropertyValue("--k")) || 0 };
      dragged = false; swipe.classList.add("dragging"); knobEl.setPointerCapture(e.pointerId);
    });
    knobEl.addEventListener("pointermove", (e) => { if (!start) return; const dx = e.clientX - start.x; if (Math.abs(dx) > 3) dragged = true; setKnob(start.k + dx, false); });
    knobEl.addEventListener("pointerup", (e) => {
      if (!start) return;
      const k = start.k + (e.clientX - start.x);
      start = null; swipe.classList.remove("dragging");
      if (!dragged) return;
      if (k / swipeMax() > 0.8) form.requestSubmit(); else setKnob(0, true);
    });
    swipe.addEventListener("click", () => { if (swipeBusy || dragged) { dragged = false; return; } form.requestSubmit(); });
    swipe.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); if (!swipeBusy) form.requestSubmit(); } });
    window.addEventListener("resize", () => { if (swipe.classList.contains("done")) setKnob(swipeMax(), false); });
  })();
  async function launch() {
    swipeBusy = true; setKnob(swipeMax(), true); swipe.classList.add("done"); swipeText.textContent = "Sending";
    if (reduceMotion) return;
    await sleep(380);
    plane.classList.add("fly");
    const r0 = knobEl.getBoundingClientRect();
    for (let i = 0; i < 7; i++) setTimeout(() => {
      const t = (i + 1) / 8, dot = document.createElement("span");
      dot.className = "plane-trail";
      dot.style.left = `${r0.left + r0.width / 2 + 340 * t * t}px`;
      dot.style.top = `${r0.top + r0.height / 2 - 260 * t * t}px`;
      document.body.appendChild(dot);
      dot.addEventListener("animationend", () => dot.remove());
    }, 120 + i * 85);
  }

  async function predict(fd) {
    if (USE_MOCK) {
      await sleep(1800);
      const text = fd.get("tweet"), a = (fd.get("aspect") || "").trim();
      const hidden = !!a && new RegExp(escapeRe(a), "i").test(text);
      return { ...MOCK_RESPONSE, text_seen: hidden ? text.replace(new RegExp(escapeRe(a), "gi"), "<mask>") : text, aspect_hidden: hidden };
    }
    let res;
    try { res = await fetch("/api/predict", { method: "POST", body: fd }); }
    catch { throw new Error("Couldn't reach the server at /api/predict. Check that the Python app is running, then try again."); }
    let data = null;
    try { data = await res.json(); } catch { /* not JSON */ }
    if (!res.ok || (data && data.error)) throw new Error((data && data.error) || `The server answered with status ${res.status}.`);
    if (!data || !data.sentiment || !data.aspect_type) throw new Error("The server's answer was missing the predictions.");
    return data;
  }

  const states = { empty: $("#stEmpty"), loading: $("#stLoading"), error: $("#stError"), done: $("#stDone") };
  const show = (name) => Object.entries(states).forEach(([k, el]) => (el.hidden = k !== name));

  const ring = (() => {
    const fg = $("#ringFg"), label = $("#ringPct"), wrap = $(".ring-wrap"), items = $$("#checklist li"), C = 326.73;
    let p = 0, target = 0, raf = 0, timers = [];
    const draw = () => { fg.style.strokeDashoffset = (C * (1 - p)).toFixed(1); label.textContent = `${Math.round(p * 100)}%`; };
    const paint = () => { p += (target - p) * 0.12; draw(); if (Math.abs(target - p) > 0.002) raf = requestAnimationFrame(paint); else { p = target; draw(); raf = 0; } };
    const go = (v) => { target = v; if (!raf) raf = requestAnimationFrame(paint); };
    const mark = (i) => items.forEach((li, k) => { li.classList.toggle("done", k < i); li.classList.toggle("active", k === i); });
    return {
      start() { timers.forEach(clearTimeout); timers = []; p = 0; target = 0; mark(0); wrap.classList.remove("complete"); draw();
        [[0, 0.18], [450, 0.42], [950, 0.66], [1450, 0.86]].forEach(([t, v], i) => timers.push(setTimeout(() => { mark(i); go(v); }, t))); },
      async finish() { timers.forEach(clearTimeout); timers = []; mark(items.length); go(1); await sleep(reduceMotion ? 0 : 450); replay(wrap, "complete"); await sleep(reduceMotion ? 0 : 450); },
      stop() { timers.forEach(clearTimeout); timers = []; },
    };
  })();

  const SENT_COLORS = { positive: "var(--pos)", neutral: "var(--neu)", negative: "var(--neg)" };
  function renderBars(el, probs, colorFor) {
    const entries = Object.entries(probs).sort((a, b) => b[1] - a[1]);
    el.innerHTML = entries.map(([k, v], i) => `
      <div class="bar-row${i === 0 ? " top" : ""}" style="--c:${colorFor(k, i)}">
        <span>${escapeHtml(k)}</span>
        <div class="bar-track"><div class="bar-fill" style="--d:${(0.25 + i * 0.1).toFixed(2)}s" data-v="${v}"></div></div>
        <span class="bar-pct" data-v="${v}">0%</span>
      </div>`).join("");
    requestAnimationFrame(() => requestAnimationFrame(() => {
      $$(".bar-fill", el).forEach((b) => (b.style.width = `${clamp(+b.dataset.v, 0, 1) * 100}%`));
      $$(".bar-pct", el).forEach((p) => countUp(p, +p.dataset.v, (v) => pct(v)));
    }));
    return entries[0];
  }
  function renderResult(data, aspect) {
    $("#seenText").innerHTML = escapeHtml(data.text_seen || "").replace(/&lt;mask&gt;/g, '<span class="mask-chip">&lt;mask&gt;</span>');
    $("#seenNote").textContent = data.aspect_hidden
      ? `"${aspect}" was in the tweet, so it was hidden. The model had to infer it from the other words and the image.`
      : aspect ? `"${aspect}" wasn't found in the tweet, so the model saw the tweet as written.` : "No aspect given, so the model saw the tweet as written.";
    const [sKey, sVal] = renderBars($("#sentBars"), data.sentiment, (k) => SENT_COLORS[k] || "var(--ink-3)");
    const sWord = $("#sentWord");
    sWord.textContent = sKey; sWord.style.setProperty("--c", SENT_COLORS[sKey] || "var(--ink)");
    $("#sentConf").textContent = `${pct(sVal)} probability`;
    const [tKey, tVal] = renderBars($("#typeBars"), data.aspect_type, (k, i) => (i === 0 ? "var(--ultra)" : "var(--ink-3)"));
    const tWord = $("#typeWord");
    tWord.textContent = tKey; tWord.style.setProperty("--c", "var(--ultra)");
    $("#typeConf").textContent = `${pct(tVal)} probability`;
    const labels = { sentiment: "Sentiment model", aspect_type: "Aspect-type model" }, models = data.models || {};
    const pct1 = (v) => (typeof v === "number" ? `${(v * 100).toFixed(1)}%` : "n/a");
    $("#modelsInfo").innerHTML = Object.entries(labels).filter(([k]) => models[k]).map(([k, label]) => {
      const m = models[k];
      return `<div class="model"><dt>${label} <span class="model-name">${escapeHtml(m.model || "")}</span></dt>
        <dd><span><strong>${pct1(m.test_acc)}</strong>test accuracy</span><span><strong>${pct1(m.macro_f1)}</strong>macro-F1</span></dd></div>`;
    }).join("");
  }

  let lastPayload = null;
  async function submit(fd, aspect) {
    hideFormError(); show("loading"); ring.start();
    dropFull.classList.add("scanning");
    const flicker = attentionFlicker(livePatchEls, 1400);
    if (innerWidth < 980) $("#results").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
    const minWait = sleep(reduceMotion ? 0 : 1700);
    try {
      const data = await predict(fd);
      await Promise.all([flicker, minWait]);
      await ring.finish();
      renderResult(data, aspect); show("done");
    } catch (err) {
      ring.stop(); $("#errMsg").textContent = err.message || String(err); show("error");
    } finally {
      dropFull.classList.remove("scanning");
      setTimeout(resetSwipe, 500);
    }
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (swipeBusy) return;
    const text = tweet.value.trim(), problems = [];
    if (!text) { problems.push("a tweet"); tweet.closest(".field").classList.add("invalid"); replay(tweet.closest(".field"), "shake"); }
    if (!imageFile) { problems.push("an image"); drop.closest(".field").classList.add("invalid"); replay(drop.closest(".field"), "shake"); }
    if (problems.length) { showFormError(`Add ${problems.join(" and ")} to get a prediction.`); setKnob(0, true); (text ? drop : tweet).focus(); return; }
    const aspect = aspectIn.value.trim(), fd = new FormData();
    fd.append("dataset", form.querySelector('input[name="dataset"]:checked').value);
    fd.append("tweet", text); fd.append("aspect", aspect); fd.append("image", imageFile, imageFile.name);
    lastPayload = { fd, aspect };
    await launch();
    submit(fd, aspect);
  });
  $("#retry").addEventListener("click", async () => { if (!lastPayload || swipeBusy) return; await launch(); submit(lastPayload.fd, lastPayload.aspect); });
}

