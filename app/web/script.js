/* ImplicitMABSA — web UI
 * Talks to the local Python server at POST /api/predict (FormData: dataset, tweet, aspect, image).
 * The mock is off: the page calls the real API. To try it without the server, open the page with ?mock
 * (for example http://127.0.0.1:5173/?mock) and it returns the sample response instead.
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
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const pct = (v) => `${Math.round(v * 100)}%`;

/** Replay a one-shot CSS animation class. */
function replay(el, cls) {
  if (!el || reduceMotion) return;
  el.classList.remove(cls);
  void el.offsetWidth;
  el.classList.add(cls);
  el.addEventListener("animationend", () => el.classList.remove(cls), { once: true });
}

/* ------------------------------------------------------------------ */
/* Pixel art, 14 x 14: one character per ViT patch                     */
/* ------------------------------------------------------------------ */
const PALETTE = {
  B: "#E9A23B", b: "#C9801E", s: "#FFF2D0", L: "#3FA34D", C: "#F7C948", M: "#6B3A26", T: "#E5484D",
  S: "#BFD8FF", W: "#FFFFFF", K: "#20233A", G: "#4CAF50", g: "#3A9142", E: "#6B4E3D", Y: "#FFD166",
};
const EXAMPLES = [
  {
    tweet: "Dinner at Joe's Diner tonight. Absolutely delicious!", aspect: "Joe's Diner", dataset: "twitter2015", file: "joes-diner.png",
    art: ["..............", ".....BBBB.....", "...BBBsBBBB...", "..BBsBBBBBsB..", ".BBBBBBsBBBBB.", ".BBsBBBBBBsBB.", ".bbbbbbbbbbbb.",
          "LLLLLLLLLLLLLL", ".CCCCCCCCCCCC.", ".MMMMMMMMMMMM.", ".MMMMMMMMMMMM.", ".TTTTTTTTTTTT.", ".bbbbbbbbbbbb.", "..BBBBBBBBBB.."],
  },
  {
    tweet: "What a performance from Arsenal tonight, the fans were unreal!", aspect: "Arsenal", dataset: "twitter2017", file: "arsenal.png",
    art: ["SSSSSSSSSSSSSS", "SSSSSWWWWSSSSS", "SSSWWWKKWWWSSS", "SSWWWKKKKWWWSS", "SSWKWWKKWWKWSS", "SWKKWWWWWWKKWS", "SWKWWWWWWWWKWS",
          "SWWWWKKKKWWWWS", "SSWWKKKKKKWWSS", "SSWWWKKKKWWWSS", "SSSWWWWWWWWSSS", "GGGGGWWWWGGGGG", "GgGGGGGGGGgGGG", "GGGgGGGGGgGGGG"],
  },
  {
    tweet: "Sunset views from Paris never get old.", aspect: "Paris", dataset: "twitter2015", file: "paris.png",
    art: ["SSSSSSESSSSSYY", "SSSSSSESSSSSYY", "SSSSSEEESSSSSS", "SSSSSESESSSSSS", "SSSSEEEEESSSSS", "SSSSSESESSSSSS", "SSSSESSSESSSSS",
          "SSSEEEEEEESSSS", "SSSESSSSSESSSS", "SSESSSSSSSESSS", "SSESSEEESSESSS", "SESSESSSESSESS", "GEGGEGGGEGGEGG", "GGGGGGGGGGGGGG"],
  },
];

/** Draw pixel art onto a canvas; each character becomes a `cell`-sized square. */
function drawArt(art, cell, voidColor) {
  const cv = document.createElement("canvas");
  cv.width = cv.height = 14 * cell;
  const ctx = cv.getContext("2d");
  art.forEach((row, r) => [...row].forEach((ch, c) => {
    ctx.fillStyle = PALETTE[ch] || voidColor;
    ctx.fillRect(c * cell, r * cell, cell, cell);
  }));
  return cv;
}

/* ------------------------------------------------------------------ */
/* Theme: pull cord lamp + eclipse reveal                              */
/* ------------------------------------------------------------------ */
const root = document.documentElement;
const isDark = () => root.dataset.theme === "dark";

function setTheme(next, x, y) {
  const apply = () => {
    root.dataset.theme = next;
    try { localStorage.setItem("imabsa-theme", next); } catch (e) { /* storage blocked */ }
    $("#cord").setAttribute("aria-label", next === "dark" ? "Switch to light theme" : "Switch to dark theme");
  };
  if (!document.startViewTransition || reduceMotion) { apply(); return; }
  const r = Math.hypot(Math.max(x, innerWidth - x), Math.max(y, innerHeight - y));
  const vt = document.startViewTransition(apply);
  vt.ready.then(() => {
    root.animate(
      { clipPath: [`circle(0px at ${x}px ${y}px)`, `circle(${r}px at ${x}px ${y}px)`] },
      { duration: 750, easing: "cubic-bezier(.2,.8,.2,1)", pseudoElement: "::view-transition-new(root)" },
    );
  });
}
const toggleTheme = (x, y) => setTheme(isDark() ? "light" : "dark", x, y);

(function cord() {
  const btn = $("#cord"), path = $("#cordPath"), knob = $("#cordKnob");
  if (!btn) return;
  btn.setAttribute("aria-label", isDark() ? "Switch to light theme" : "Switch to dark theme");
  const REST = 70;
  let len = REST, sx = 0, vy = 0, vx = 0, raf = 0, drag = null, justDragged = false;

  function render() {
    path.setAttribute("d", `M20 0 Q ${20 + sx * 0.4} ${len * 0.55} ${20 + sx} ${len}`);
    knob.style.transform = `translate(${sx}px, ${len - 6}px)`;
  }
  function spring() {
    // damped spring back to rest, with a little sideways sway
    vy += (REST - len) * 0.18; vy *= 0.78;
    vx += -sx * 0.12; vx *= 0.86;
    len += vy; sx += vx;
    render();
    if (Math.abs(vy) + Math.abs(len - REST) + Math.abs(vx) + Math.abs(sx) > 0.15) raf = requestAnimationFrame(spring);
    else { len = REST; sx = 0; render(); raf = 0; }
  }
  const release = () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(spring); };
  const knobPoint = () => { const r = knob.getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2]; };

  btn.addEventListener("pointerdown", (e) => {
    cancelAnimationFrame(raf);
    drag = { y0: e.clientY, x0: e.clientX, moved: 0 };
    btn.setPointerCapture(e.pointerId);
  });
  btn.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const dy = e.clientY - drag.y0, dx = e.clientX - drag.x0;
    drag.moved = Math.max(drag.moved, Math.hypot(dx, dy));
    len = REST + clamp(dy, -10, 90) * 0.85;
    sx = clamp(dx, -40, 40) * 0.5;
    render();
  });
  btn.addEventListener("pointerup", () => {
    if (!drag) return;
    const pulled = len - REST, moved = drag.moved;
    drag = null;
    if (moved > 4) justDragged = true;
    if (pulled > 28) { const [x, y] = knobPoint(); toggleTheme(x, y); }
    vy = pulled > 28 ? -6 : 0;
    release();
  });
  btn.addEventListener("click", () => {
    if (justDragged) { justDragged = false; return; }
    // keyboard or plain click: play a short pull, then switch
    len = REST + 38; render();
    const [x, y] = knobPoint();
    toggleTheme(x, y);
    vy = -4; release();
  });
  render();
})();

/* ------------------------------------------------------------------ */
/* Header: border on scroll, gooey tab bar with scroll spy, menu morph */
/* ------------------------------------------------------------------ */
(function header() {
  const bar = $(".bar");
  const onScroll = () => bar.classList.toggle("scrolled", window.scrollY > 8);
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  const nav = $("#nav");
  const links = $$("a[data-spy]", nav);
  const blobs = $$(".blob", nav);
  let current = null;
  function moveTo(link) {
    links.forEach((l) => l.classList.toggle("active", l === link));
    blobs.forEach((b) => {
      if (!link) { b.style.setProperty("--o", 0); return; }
      b.style.setProperty("--x", `${link.offsetLeft}px`);
      b.style.setProperty("--w", `${link.offsetWidth}px`);
      b.style.setProperty("--o", 1);
    });
  }
  const io = new IntersectionObserver((entries) => {
    entries.forEach((en) => {
      if (en.isIntersecting) current = links.find((l) => l.dataset.spy === en.target.id) || null;
      else if (current && current.dataset.spy === en.target.id) current = null;
    });
    moveTo(current);
  }, { rootMargin: "-45% 0px -50% 0px" });
  ["how", "lab", "findings"].forEach((id) => io.observe(document.getElementById(id)));
  window.addEventListener("resize", () => moveTo(current));

  const menuBtn = $("#menuBtn"), menu = $("#mobileMenu");
  const setMenu = (open) => { menu.hidden = !open; menuBtn.setAttribute("aria-expanded", String(open)); menuBtn.setAttribute("aria-label", open ? "Close menu" : "Open menu"); };
  menuBtn.addEventListener("click", () => setMenu(menu.hidden));
  menu.addEventListener("click", (e) => { if (e.target.closest("a")) setMenu(false); });
  $("#mmTheme").addEventListener("click", () => {
    const r = menuBtn.getBoundingClientRect();
    toggleTheme(r.left + r.width / 2, r.top + r.height / 2);
  });
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
      el.style.transition = "transform .15s linear, box-shadow .3s, background .2s, color .2s";
      el.style.transform = `translate(${dx * 0.28}px, ${dy * 0.38}px)`;
      if (inner) inner.style.transform = `translate(${dx * 0.12}px, ${dy * 0.16}px)`;
    });
    el.addEventListener("pointerleave", () => {
      el.style.transition = "transform .7s var(--spring), box-shadow .3s, background .2s, color .2s";
      el.style.transform = "";
      if (inner) inner.style.transform = "";
    });
    el.addEventListener("pointerdown", () => replay(inner || el, "jelly"));
  });
})();

/* ------------------------------------------------------------------ */
/* Hero headline: the word gets masked, then becomes another word      */
/* ------------------------------------------------------------------ */
(function heroMask() {
  const el = $("#heroMask");
  if (!el) return;
  const words = ["word", "brand", "place", "team", "person"];
  if (reduceMotion) { el.textContent = "<mask>"; el.classList.add("is-masked"); return; }
  let i = 0;
  const scramble = async (target) => {
    const glyphs = "abcdefghijklmnopqrstuvwxyz";
    for (let f = 0; f < 8; f++) {
      el.textContent = target.split("").map((c, k) => (k < f - 2 ? c : glyphs[(Math.random() * 26) | 0])).join("");
      await sleep(45);
    }
    el.textContent = target;
  };
  (async function loop() {
    await sleep(1800);
    while (true) {
      el.classList.add("is-masked");
      el.textContent = "<mask>";
      await sleep(1500);
      i = (i + 1) % words.length;
      el.classList.remove("is-masked");
      await scramble(words[i]);
      await sleep(2000);
    }
  })();
})();

/* ------------------------------------------------------------------ */
/* Patch grids                                                         */
/* ------------------------------------------------------------------ */
const GRID = 14;

/** Make 196 patch divs. `paint(div, row, col)` styles each one. */
function buildPatches(container, paint) {
  container.innerHTML = "";
  const frag = document.createDocumentFragment();
  const c = (GRID - 1) / 2;
  for (let r = 0; r < GRID; r++) {
    for (let col = 0; col < GRID; col++) {
      const p = document.createElement("div");
      p.className = "patch";
      const ang = Math.random() * Math.PI * 2;
      const dist = 400 + Math.random() * 900;
      p.style.setProperty("--sx", `${Math.cos(ang) * dist}%`);
      p.style.setProperty("--sy", `${Math.sin(ang) * dist}%`);
      p.style.setProperty("--sr", `${(Math.random() - 0.5) * 220}deg`);
      p.style.setProperty("--d", `${(Math.hypot(r - c, col - c) * 0.045 + Math.random() * 0.12).toFixed(3)}s`);
      paint(p, r, col);
      frag.appendChild(p);
    }
  }
  container.appendChild(frag);
  return [...container.children];
}

/** Scatter patches, then let them fly back into place. */
async function assemble(container) {
  container.classList.remove("gridded");
  container.classList.add("scatter");
  if (reduceMotion) { container.classList.remove("scatter"); return; }
  void container.offsetWidth;
  await sleep(30);
  container.classList.remove("scatter");
  await sleep(1500);
}

/** Light up random patches for a moment, like attention moving over the image. */
function attentionFlicker(patches, ms) {
  if (reduceMotion || !patches.length) return Promise.resolve();
  return new Promise((resolve) => {
    const end = performance.now() + ms;
    const lit = new Set();
    (function step() {
      lit.forEach((p) => p.classList.remove("hot"));
      lit.clear();
      if (performance.now() > end) return resolve();
      const n = 6 + ((Math.random() * 8) | 0);
      for (let k = 0; k < n; k++) {
        const p = patches[(Math.random() * patches.length) | 0];
        p.classList.add("hot"); lit.add(p);
      }
      setTimeout(step, 140);
    })();
  });
}

/* ------------------------------------------------------------------ */
/* Hero demo: type a tweet, mask the aspect, assemble the photo, predict */
/* ------------------------------------------------------------------ */
(function demo() {
  const tweetEl = $("#demoTweet");
  const patchesEl = $("#demoPatches");
  const photo = $(".demo-photo");
  const out = $("#demoOut");
  const card = $(".demo-card");
  if (!tweetEl) return;

  const ART = EXAMPLES[0].art;
  const patches = buildPatches(patchesEl, (p, r, c) => {
    const col = PALETTE[ART[r][c]];
    if (col) p.style.background = col; else p.classList.add("patch-void");
  });

  const before = "Dinner at ", aspect = "Joe's Diner", after = " tonight. Absolutely delicious!";
  const bars = $$(".demo-row b", out);

  if (reduceMotion) {
    tweetEl.innerHTML = `${escapeHtml(before)}<span class="mask-chip">&lt;mask&gt;</span>${escapeHtml(after)}`;
    bars.forEach((b) => (b.style.width = `${b.dataset.v * 100}%`));
    out.classList.add("show");
    patchesEl.classList.add("gridded");
    return;
  }

  async function type(text, el) {
    for (const ch of text) {
      el.insertAdjacentText("beforeend", ch);
      await sleep(ch === " " ? 40 : 28 + Math.random() * 40);
    }
  }

  (async function run() {
    while (true) {
      out.classList.remove("show");
      bars.forEach((b) => (b.style.width = "0"));
      tweetEl.innerHTML = "";
      patchesEl.classList.remove("gridded");
      patchesEl.classList.add("scatter");
      const textNode = document.createElement("span");
      const caret = document.createElement("span");
      caret.className = "caret";
      tweetEl.append(textNode, caret);

      await sleep(400);
      await type(before, textNode);
      const asp = document.createElement("span");
      asp.className = "aspect-span";
      textNode.appendChild(asp);
      await type(aspect, asp);
      const rest = document.createElement("span");
      textNode.appendChild(rest);
      await type(after, rest);
      caret.remove();

      const assembled = assemble(patchesEl);
      await sleep(500);
      asp.classList.add("is-target");
      await sleep(900);
      asp.outerHTML = '<span class="mask-chip">&lt;mask&gt;</span>';
      await assembled;

      patchesEl.classList.add("gridded");
      photo.classList.add("scanning");
      await attentionFlicker(patches, 1800);
      photo.classList.remove("scanning");

      bars.forEach((b, i) => setTimeout(() => (b.style.width = `${b.dataset.v * 100}%`), i * 120));
      out.classList.add("show");
      await sleep(4200);
    }
  })();

  const fig = $(".demo");
  fig.addEventListener("pointermove", (e) => {
    const r = fig.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width - 0.5, y = (e.clientY - r.top) / r.height - 0.5;
    card.style.transform = `rotateY(${x * 10}deg) rotateX(${-y * 10}deg)`;
  });
  fig.addEventListener("pointerleave", () => (card.style.transform = ""));
})();

/* "Read the photo" step: a small 14 x 14 grid that pings */
(function stepPatches() {
  const el = $(".v-patch");
  if (!el) return;
  const c = (GRID - 1) / 2;
  for (let r = 0; r < GRID; r++) for (let col = 0; col < GRID; col++) {
    const i = document.createElement("i");
    i.style.setProperty("--d", `${(Math.hypot(r - c, col - c) * 0.18).toFixed(2)}s`);
    el.appendChild(i);
  }
})();

/* ------------------------------------------------------------------ */
/* Elastic string above the four steps: pluck it with the pointer      */
/* ------------------------------------------------------------------ */
(function elasticString() {
  const wrap = $(".steps-wrap"), svg = $("#string"), path = $("#stringPath");
  if (!wrap || reduceMotion) { if (svg) svg.style.display = "none"; return; }
  const BASE = 40;
  let W = 0, cx = 0, disp = 0, vel = 0, grabbed = false, raf = 0;
  const NODES = 4;
  const dots = Array.from({ length: NODES }, () => {
    const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    c.setAttribute("r", "5");
    c.setAttribute("fill", "var(--ultra)");
    svg.appendChild(c);
    return c;
  });

  function size() {
    W = wrap.clientWidth;
    svg.setAttribute("viewBox", `0 0 ${W} 80`);
    if (!cx) cx = W / 2;
    draw();
  }
  function curveY(x) {
    // quadratic through (0,BASE), control (cx, BASE + 2*disp), (W,BASE): solve x(t) numerically
    let lo = 0, hi = 1;
    for (let k = 0; k < 18; k++) {
      const t = (lo + hi) / 2;
      const xt = 2 * (1 - t) * t * cx + t * t * W;
      if (xt < x) lo = t; else hi = t;
    }
    const t = (lo + hi) / 2;
    return BASE + 2 * (1 - t) * t * (2 * disp);
  }
  function draw() {
    path.setAttribute("d", `M0 ${BASE} Q ${cx} ${BASE + 2 * disp} ${W} ${BASE}`);
    dots.forEach((d, i) => {
      const x = (W / NODES) * (i + 0.5);
      d.setAttribute("cx", x);
      d.setAttribute("cy", curveY(x));
    });
  }
  function spring() {
    vel += -disp * 0.16; vel *= 0.9; disp += vel;
    draw();
    if (Math.abs(disp) + Math.abs(vel) > 0.05) raf = requestAnimationFrame(spring);
    else { disp = 0; vel = 0; draw(); raf = 0; }
  }
  const letGo = () => { if (grabbed) { grabbed = false; cancelAnimationFrame(raf); raf = requestAnimationFrame(spring); } };

  // the line sits 36px above the cards: listen on the wrapper, measured in svg coordinates
  wrap.addEventListener("pointermove", (e) => {
    const r = svg.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top - BASE;
    if (!grabbed && Math.abs(y) < 12 && Math.abs(y - disp) < 14) { grabbed = true; cancelAnimationFrame(raf); }
    if (grabbed) {
      cx = clamp(x, W * 0.1, W * 0.9);
      disp = clamp(y, -36, 36);
      draw();
      if (Math.abs(y) >= 36) letGo();
    }
  });
  wrap.addEventListener("pointerleave", letGo);
  window.addEventListener("resize", size);
  size();
  // a tiny pluck when the section first appears, so people notice it
  new IntersectionObserver((en, obs) => {
    if (en[0].isIntersecting) { disp = 18; vel = 0; cancelAnimationFrame(raf); raf = requestAnimationFrame(spring); obs.disconnect(); }
  }, { threshold: 0.6 }).observe(wrap);
})();

/* ------------------------------------------------------------------ */
/* Lab form                                                            */
/* ------------------------------------------------------------------ */
const form = $("#form");
const tweet = $("#tweet");
const aspectIn = $("#aspect");
const mirror = $("#tweetMirror");
const maskHint = $("#maskHint");
const count = $("#count");
const fileIn = $("#image");
const drop = $("#drop");
const dropEmpty = $("#dropEmpty");
const dropFull = $("#dropFull");
const livePatches = $("#livePatches");
const fileName = $("#fileName");
const clearImg = $("#clearImg");
const formError = $("#formError");
const seg = $("#seg");

let imageFile = null;
let livePatchEls = [];

/* Stretch toggle: the leading edge moves first, the trailing edge catches up. */
(function stretchToggle() {
  const sync = (dir) => {
    const v = form.querySelector('input[name="dataset"]:checked').value;
    seg.classList.toggle("go-right", dir === "right");
    seg.classList.toggle("go-left", dir === "left");
    seg.classList.toggle("at-2017", v === "twitter2017");
  };
  seg.addEventListener("change", (e) => sync(e.target.value === "twitter2017" ? "right" : "left"));
  sync();
})();
function setDataset(v) {
  const input = form.querySelector(`input[name="dataset"][value="${v}"]`);
  if (input && !input.checked) { input.checked = true; input.dispatchEvent(new Event("change", { bubbles: true })); }
}

/* Show which words will be hidden, live, under the textarea. */
function updateMirror() {
  const text = tweet.value;
  const a = aspectIn.value.trim();
  count.textContent = `${text.length} / 400`;
  let found = false;
  let html = escapeHtml(text);
  if (a) {
    const re = new RegExp(escapeRe(escapeHtml(a)), "gi");
    html = html.replace(re, (m) => { found = true; return `<mark>${m}</mark>`; });
  }
  mirror.innerHTML = html + "\n";
  mirror.scrollTop = tweet.scrollTop;
  maskHint.classList.toggle("on", found);
  if (!a) maskHint.textContent = "Name an aspect to hide it from the model.";
  else if (found) maskHint.textContent = `"${a}" is in the tweet, so the model will see <mask> instead.`;
  else maskHint.textContent = `"${a}" isn't in the tweet, so the model sees the tweet as written.`;
}
tweet.addEventListener("input", () => { updateMirror(); tweet.closest(".field").classList.remove("invalid"); hideFormError(); });
tweet.addEventListener("scroll", () => (mirror.scrollTop = tweet.scrollTop));
aspectIn.addEventListener("input", updateMirror);
updateMirror();

/* Image: crop to a square, then show it as 196 patches flying into place. */
function loadImage(file) {
  if (!file) return;
  if (!/^image\/(png|jpe?g)$/i.test(file.type)) {
    showFormError("Choose a JPG or PNG image.");
    return;
  }
  hideFormError();
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    const S = 448;
    const cv = document.createElement("canvas");
    cv.width = cv.height = S;
    const side = Math.min(img.naturalWidth, img.naturalHeight);
    cv.getContext("2d").drawImage(img, (img.naturalWidth - side) / 2, (img.naturalHeight - side) / 2, side, side, 0, 0, S, S);
    URL.revokeObjectURL(url);
    const data = cv.toDataURL("image/jpeg", 0.9);
    imageFile = file;
    drop.classList.add("has-img");
    drop.closest(".field").classList.remove("invalid");
    dropEmpty.hidden = true;
    dropFull.hidden = false;
    fileName.textContent = file.name;
    clearImg.hidden = false;
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
  imageFile = null;
  fileIn.value = "";
  livePatches.innerHTML = "";
  livePatchEls = [];
  livePatches.classList.remove("gridded");
  drop.classList.remove("has-img");
  dropEmpty.hidden = false;
  dropFull.hidden = true;
  fileName.textContent = "";
  clearImg.hidden = true;
});

function showFormError(msg) { formError.textContent = msg; formError.hidden = false; }
function hideFormError() { formError.hidden = true; }

/* Card fan of examples: click one to fill the whole form. */
(function fan() {
  const fanEl = $("#fan");
  EXAMPLES.forEach((ex) => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "fan-card";
    card.setAttribute("role", "listitem");
    card.setAttribute("aria-label", `Use example: ${ex.tweet}`);
    const thumb = drawArt(ex.art, 1, "#F1F2F7");
    const txt = document.createElement("span");
    txt.className = "fc-text";
    txt.textContent = ex.tweet;
    const asp = document.createElement("span");
    asp.className = "fc-asp";
    asp.innerHTML = `aspect <b>${escapeHtml(ex.aspect)}</b>`;
    card.append(thumb, txt, asp);
    card.addEventListener("click", () => {
      replay(card, "picked");
      tweet.value = ex.tweet;
      aspectIn.value = ex.aspect;
      setDataset(ex.dataset);
      updateMirror();
      tweet.closest(".field").classList.remove("invalid");
      drawArt(ex.art, 32, "#F1F2F7").toBlob((blob) => loadImage(new File([blob], ex.file, { type: "image/png" })), "image/png");
      const r = form.getBoundingClientRect();
      if (r.top < 0 || r.top > innerHeight * 0.6) form.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
    });
    fanEl.appendChild(card);
  });
})();

/* ------------------------------------------------------------------ */
/* Swipe to confirm + paper plane                                      */
/* ------------------------------------------------------------------ */
const swipe = $("#swipe"), knobEl = $("#swipeKnob"), swipeText = $("#swipeText"), plane = $("#plane");
const swipeMax = () => swipe.clientWidth - 64;
let swipeBusy = false;

function setKnob(px, animate) {
  swipe.classList.toggle("settle", !!animate);
  const m = swipeMax();
  swipe.style.setProperty("--k", `${clamp(px, 0, m)}px`);
  swipe.style.setProperty("--kp", (clamp(px, 0, m) / m).toFixed(3));
}
function resetSwipe() {
  swipe.classList.remove("done");
  swipeText.textContent = "Slide to predict";
  plane.classList.remove("fly");
  setKnob(0, true);
  swipeBusy = false;
}
(function swipeDrag() {
  let start = null, dragged = false;
  knobEl.addEventListener("pointerdown", (e) => {
    if (swipeBusy) return;
    start = { x: e.clientX, k: parseFloat(getComputedStyle(swipe).getPropertyValue("--k")) || 0 };
    dragged = false;
    swipe.classList.add("dragging");
    knobEl.setPointerCapture(e.pointerId);
  });
  knobEl.addEventListener("pointermove", (e) => {
    if (!start) return;
    const dx = e.clientX - start.x;
    if (Math.abs(dx) > 3) dragged = true;
    setKnob(start.k + dx, false);
  });
  knobEl.addEventListener("pointerup", (e) => {
    if (!start) return;
    const k = start.k + (e.clientX - start.x);
    start = null;
    swipe.classList.remove("dragging");
    if (!dragged) return; // a plain click is handled by the swipe's click listener
    if (k / swipeMax() > 0.8) form.requestSubmit();
    else setKnob(0, true);
  });
  swipe.addEventListener("click", (e) => {
    if (swipeBusy || dragged) { dragged = false; return; }
    form.requestSubmit();
  });
  swipe.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); if (!swipeBusy) form.requestSubmit(); }
  });
  window.addEventListener("resize", () => { if (swipe.classList.contains("done")) setKnob(swipeMax(), false); });
})();

/** Slide the knob home and send the paper plane off. */
async function launch() {
  swipeBusy = true;
  setKnob(swipeMax(), true);
  swipe.classList.add("done");
  swipeText.textContent = "Sending";
  if (reduceMotion) return;
  await sleep(380);
  plane.classList.add("fly");
  // a short dotted trail behind the plane
  const r0 = knobEl.getBoundingClientRect();
  for (let i = 0; i < 7; i++) {
    setTimeout(() => {
      const t = (i + 1) / 8;
      const dot = document.createElement("span");
      dot.className = "plane-trail";
      dot.style.position = "fixed";
      dot.style.left = `${r0.left + r0.width / 2 + 340 * t * t}px`;
      dot.style.top = `${r0.top + r0.height / 2 - 260 * t * t}px`;
      document.body.appendChild(dot);
      dot.addEventListener("animationend", () => dot.remove());
    }, 120 + i * 85);
  }
}

/* ------------------------------------------------------------------ */
/* Prediction                                                          */
/* ------------------------------------------------------------------ */
async function predict(fd) {
  if (USE_MOCK) {
    await sleep(1800);
    const text = fd.get("tweet"), a = (fd.get("aspect") || "").trim();
    const hidden = !!a && new RegExp(escapeRe(a), "i").test(text);
    return {
      ...MOCK_RESPONSE,
      text_seen: hidden ? text.replace(new RegExp(escapeRe(a), "gi"), "<mask>") : text,
      aspect_hidden: hidden,
    };
  }
  let res;
  try {
    res = await fetch("/api/predict", { method: "POST", body: fd });
  } catch {
    throw new Error("Couldn't reach the server at /api/predict. Check that the Python app is running, then try again.");
  }
  let data = null;
  try { data = await res.json(); } catch { /* not JSON */ }
  if (!res.ok || (data && data.error)) {
    throw new Error((data && data.error) || `The server answered with status ${res.status}.`);
  }
  if (!data || !data.sentiment || !data.aspect_type) throw new Error("The server's answer was missing the predictions.");
  return data;
}

const states = { empty: $("#stEmpty"), loading: $("#stLoading"), error: $("#stError"), done: $("#stDone") };
function show(name) { Object.entries(states).forEach(([k, el]) => (el.hidden = k !== name)); }

/* Progress ring + checklist shown while the request runs. */
const ring = (() => {
  const fg = $("#ringFg"), label = $("#ringPct"), wrap = $(".ring-wrap"), items = $$("#checklist li");
  const C = 326.73;
  let p = 0, target = 0, raf = 0, timers = [];
  const paint = () => {
    p += (target - p) * 0.12;
    fg.style.strokeDashoffset = (C * (1 - p)).toFixed(1);
    label.textContent = `${Math.round(p * 100)}%`;
    if (Math.abs(target - p) > 0.002) raf = requestAnimationFrame(paint);
    else { p = target; fg.style.strokeDashoffset = (C * (1 - p)).toFixed(1); label.textContent = `${Math.round(p * 100)}%`; raf = 0; }
  };
  const go = (v) => { target = v; if (!raf) raf = requestAnimationFrame(paint); };
  const mark = (i) => items.forEach((li, k) => { li.classList.toggle("done", k < i); li.classList.toggle("active", k === i); });
  return {
    start() {
      timers.forEach(clearTimeout); timers = [];
      p = 0; target = 0; mark(0); wrap.classList.remove("complete");
      fg.style.strokeDashoffset = C; label.textContent = "0%";
      [[0, 0.18], [450, 0.42], [950, 0.66], [1450, 0.86]].forEach(([t, v], i) => timers.push(setTimeout(() => { mark(i); go(v); }, t)));
    },
    async finish() {
      timers.forEach(clearTimeout); timers = [];
      mark(items.length); go(1);
      await sleep(reduceMotion ? 0 : 450);
      replay(wrap, "complete");
      await sleep(reduceMotion ? 0 : 450);
    },
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

function countUp(el, v, fmt, dur = 1000) {
  if (reduceMotion) { el.textContent = fmt(v); return; }
  const t0 = performance.now();
  (function f(now) {
    const t = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - t, 3);
    el.textContent = fmt(v * e);
    if (t < 1) requestAnimationFrame(f);
  })(t0);
}

function renderResult(data, aspect) {
  $("#seenText").innerHTML = escapeHtml(data.text_seen || "").replace(/&lt;mask&gt;/g, '<span class="mask-chip">&lt;mask&gt;</span>');
  $("#seenNote").textContent = data.aspect_hidden
    ? `"${aspect}" was in the tweet, so it was hidden. The model had to infer it from the other words and the image.`
    : aspect
      ? `"${aspect}" wasn't found in the tweet, so the model saw the tweet as written.`
      : "No aspect given, so the model saw the tweet as written.";

  const [sKey, sVal] = renderBars($("#sentBars"), data.sentiment, (k) => SENT_COLORS[k] || "var(--ink-3)");
  const sWord = $("#sentWord");
  sWord.textContent = sKey;
  sWord.style.setProperty("--c", SENT_COLORS[sKey] || "var(--ink)");
  $("#sentConf").textContent = `${pct(sVal)} probability`;

  const [tKey, tVal] = renderBars($("#typeBars"), data.aspect_type, (k, i) => (i === 0 ? "var(--ultra)" : "var(--ink-3)"));
  const tWord = $("#typeWord");
  tWord.textContent = tKey;
  tWord.style.setProperty("--c", "var(--ultra)");
  $("#typeConf").textContent = `${pct(tVal)} probability`;

  const labels = { sentiment: "Sentiment model", aspect_type: "Aspect-type model" };
  const models = data.models || {};
  const pct1 = (v) => (typeof v === "number" ? `${(v * 100).toFixed(1)}%` : "n/a");
  $("#modelsInfo").innerHTML = Object.entries(labels).filter(([k]) => models[k]).map(([k, label]) => {
    const m = models[k];
    return `<div class="model">
      <dt>${label} <span class="model-name">${escapeHtml(m.model || "")}</span></dt>
      <dd><span><strong>${pct1(m.test_acc)}</strong>test accuracy</span><span><strong>${pct1(m.macro_f1)}</strong>macro-F1</span></dd>
    </div>`;
  }).join("");
}

let lastPayload = null;

async function submit(fd, aspect) {
  hideFormError();
  show("loading");
  ring.start();
  dropFull.classList.add("scanning");
  const flicker = attentionFlicker(livePatchEls, 1400);
  if (window.innerWidth < 980) $("#results").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
  const minWait = sleep(reduceMotion ? 0 : 1700);
  try {
    const data = await predict(fd);
    await Promise.all([flicker, minWait]);
    await ring.finish();
    renderResult(data, aspect);
    show("done");
  } catch (err) {
    ring.stop();
    $("#errMsg").textContent = err.message || String(err);
    show("error");
  } finally {
    dropFull.classList.remove("scanning");
    setTimeout(resetSwipe, 500);
  }
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (swipeBusy) return;
  const text = tweet.value.trim();
  const problems = [];
  if (!text) { problems.push("a tweet"); replay(tweet.closest(".field"), "invalid"); tweet.closest(".field").classList.add("invalid"); }
  if (!imageFile) { problems.push("an image"); drop.closest(".field").classList.add("invalid"); }
  if (problems.length) {
    showFormError(`Add ${problems.join(" and ")} to get a prediction.`);
    setKnob(0, true);
    (text ? drop : tweet).focus();
    return;
  }
  const aspect = aspectIn.value.trim();
  const fd = new FormData();
  fd.append("dataset", form.querySelector('input[name="dataset"]:checked').value);
  fd.append("tweet", text);
  fd.append("aspect", aspect);
  fd.append("image", imageFile, imageFile.name);
  lastPayload = { fd, aspect };
  await launch();
  submit(fd, aspect);
});

$("#retry").addEventListener("click", async () => {
  if (!lastPayload || swipeBusy) return;
  await launch();
  submit(lastPayload.fd, lastPayload.aspect);
});

/* ------------------------------------------------------------------ */
/* Findings: count the numbers up when they scroll into view           */
/* ------------------------------------------------------------------ */
(function facts() {
  const els = $$(".facts strong[data-count]");
  const io = new IntersectionObserver((entries) => {
    entries.forEach((en) => {
      if (!en.isIntersecting) return;
      const el = en.target, v = +el.dataset.count, dec = +el.dataset.dec, suf = el.dataset.suffix || "";
      countUp(el, v, (x) => `${x.toFixed(dec)}${suf}`, 1400);
      io.unobserve(el);
    });
  }, { threshold: 0.6 });
  els.forEach((el) => io.observe(el));
})();
