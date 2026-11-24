/* ImplicitMABSA — web UI
 * Talks to the local Python server at POST /api/predict (FormData: dataset, tweet, aspect, image).
 * Set USE_MOCK = true to try the page without the server; keep it false in the repository.
 */
const USE_MOCK = false;

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
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const pct = (v) => `${Math.round(v * 100)}%`;

/* ------------------------------------------------------------------ */
/* Ambient field: a quiet grid of patches that light up near the cursor */
/* ------------------------------------------------------------------ */
(function field() {
  const canvas = $("#field");
  if (!canvas || reduceMotion) return;
  const ctx = canvas.getContext("2d");
  const GAP = 30, SIZE = 3, RADIUS = 170;
  let w = 0, h = 0, dpr = 1, mx = -9999, my = -9999, tx = -9999, ty = -9999, raf = 0, energy = 0;

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    w = window.innerWidth; h = window.innerHeight;
    canvas.width = w * dpr; canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }
  function draw() {
    ctx.clearRect(0, 0, w, h);
    const ox = (w % GAP) / 2, oy = (h % GAP) / 2;
    for (let y = oy; y < h; y += GAP) {
      for (let x = ox; x < w; x += GAP) {
        const d = Math.hypot(x - mx, y - my);
        const t = Math.max(0, 1 - d / RADIUS) * energy;
        const s = SIZE + t * 9;
        ctx.fillStyle = t > 0.02 ? `rgba(61,43,255,${0.12 + t * 0.55})` : "rgba(20,23,43,0.07)";
        ctx.fillRect(x - s / 2, y - s / 2, s, s);
      }
    }
  }
  function tick() {
    mx += (tx - mx) * 0.14; my += (ty - my) * 0.14;
    energy += ((tx > -999 ? 1 : 0) - energy) * 0.06;
    draw();
    if (Math.abs(tx - mx) > 0.5 || Math.abs(ty - my) > 0.5 || (energy > 0.01 && energy < 0.99)) raf = requestAnimationFrame(tick);
    else raf = 0;
  }
  const kick = () => { if (!raf) raf = requestAnimationFrame(tick); };
  window.addEventListener("pointermove", (e) => { tx = e.clientX; ty = e.clientY; if (mx < -999) { mx = tx; my = ty; } kick(); }, { passive: true });
  document.addEventListener("pointerleave", () => { tx = ty = -9999; kick(); });
  window.addEventListener("resize", resize);
  resize();
})();

/* header border once the page scrolls */
(function bar() {
  const el = $(".bar");
  const on = () => el.classList.toggle("scrolled", window.scrollY > 8);
  window.addEventListener("scroll", on, { passive: true });
  on();
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
  void container.offsetWidth; // commit the scattered state
  await sleep(30);
  container.classList.remove("scatter");
  await sleep(1500);
}

/** Light up random patches for a moment, like attention moving over the image. */
function attentionFlicker(patches, ms) {
  if (reduceMotion) return Promise.resolve();
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

  // 14 x 14 pixel-art burger, one character per patch
  const ART = [
    "..............",
    ".....BBBB.....",
    "...BBBsBBBB...",
    "..BBsBBBBBsB..",
    ".BBBBBBsBBBBB.",
    ".BBsBBBBBBsBB.",
    ".bbbbbbbbbbbb.",
    "LLLLLLLLLLLLLL",
    ".CCCCCCCCCCCC.",
    ".MMMMMMMMMMMM.",
    ".MMMMMMMMMMMM.",
    ".TTTTTTTTTTTT.",
    ".bbbbbbbbbbbb.",
    "..BBBBBBBBBB..",
  ];
  const COLORS = { ".": "#F1F2F7", B: "#E9A23B", b: "#C9801E", s: "#FFF2D0", L: "#3FA34D", C: "#F7C948", M: "#6B3A26", T: "#E5484D" };
  const patches = buildPatches(patchesEl, (p, r, c) => { p.style.background = COLORS[ART[r][c]] || COLORS["."]; });

  const before = "Dinner at ", aspect = "Joe's Diner", after = " tonight. Absolutely delicious!";
  const bars = [...out.querySelectorAll(".demo-row b")];

  function finalState() {
    tweetEl.innerHTML = `${escapeHtml(before)}<span class="mask-chip">&lt;mask&gt;</span>${escapeHtml(after)}`;
    bars.forEach((b) => (b.style.width = `${b.dataset.v * 100}%`));
    out.classList.add("show");
    patchesEl.classList.add("gridded");
  }
  if (reduceMotion) { finalState(); return; }

  async function type(text, el) {
    for (const ch of text) {
      el.insertAdjacentText("beforeend", ch);
      await sleep(ch === " " ? 40 : 28 + Math.random() * 40);
    }
  }

  async function run() {
    while (true) {
      // reset
      out.classList.remove("show");
      bars.forEach((b) => (b.style.width = "0"));
      tweetEl.innerHTML = "";
      patchesEl.classList.remove("gridded");
      patchesEl.classList.add("scatter");
      const textNode = document.createElement("span");
      const caret = document.createElement("span");
      caret.className = "caret";
      tweetEl.append(textNode, caret);

      // type the tweet, with the aspect in its own span
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

      // the photo flies together while the aspect is found and masked
      const assembled = assemble(patchesEl);
      await sleep(500);
      asp.classList.add("is-target");
      await sleep(900);
      asp.outerHTML = '<span class="mask-chip">&lt;mask&gt;</span>';
      await assembled;

      // read the patches
      patchesEl.classList.add("gridded");
      photo.classList.add("scanning");
      await attentionFlicker(patches, 1800);
      photo.classList.remove("scanning");

      // predict
      bars.forEach((b, i) => setTimeout(() => (b.style.width = `${b.dataset.v * 100}%`), i * 120));
      out.classList.add("show");
      await sleep(4200);
    }
  }
  run();

  // gentle 3D tilt that follows the pointer
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
const go = $("#go");
const formError = $("#formError");

let imageFile = null;
let livePatchEls = [];

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
tweet.addEventListener("input", () => { updateMirror(); tweet.closest(".field").classList.remove("invalid"); });
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
drop.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files[0];
  if (f) loadImage(f);
});
clearImg.addEventListener("click", () => {
  imageFile = null;
  fileIn.value = "";
  livePatches.innerHTML = "";
  livePatches.classList.remove("gridded");
  drop.classList.remove("has-img");
  dropEmpty.hidden = false;
  dropFull.hidden = true;
  fileName.textContent = "";
  clearImg.hidden = true;
});

function showFormError(msg) { formError.textContent = msg; formError.hidden = false; }
function hideFormError() { formError.hidden = true; }

/* ------------------------------------------------------------------ */
/* Prediction                                                          */
/* ------------------------------------------------------------------ */
async function predict(fd) {
  if (USE_MOCK) {
    await sleep(1600);
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

const SENT_COLORS = { positive: "var(--pos)", neutral: "var(--neu)", negative: "var(--neg)" };

function renderBars(el, probs, colorFor) {
  const entries = Object.entries(probs).sort((a, b) => b[1] - a[1]);
  el.innerHTML = entries.map(([k, v], i) => `
    <div class="bar-row${i === 0 ? " top" : ""}" style="--c:${colorFor(k, i)}">
      <span>${escapeHtml(k)}</span>
      <div class="bar-track"><div class="bar-fill" style="--d:${(0.15 + i * 0.1).toFixed(2)}s" data-v="${v}"></div></div>
      <span class="bar-pct" data-v="${v}">0%</span>
    </div>`).join("");
  requestAnimationFrame(() => requestAnimationFrame(() => {
    el.querySelectorAll(".bar-fill").forEach((b) => (b.style.width = `${Math.max(0, Math.min(1, b.dataset.v)) * 100}%`));
    el.querySelectorAll(".bar-pct").forEach((p) => countUp(p, +p.dataset.v));
  }));
  return entries[0];
}

function countUp(el, v) {
  if (reduceMotion) { el.textContent = pct(v); return; }
  const t0 = performance.now(), dur = 1000;
  (function f(now) {
    const t = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - t, 3);
    el.textContent = pct(v * e);
    if (t < 1) requestAnimationFrame(f);
  })(t0);
}

function renderResult(data, aspect) {
  // the text the model actually received, with the mask token drawn as a chip
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
  $("#modelsInfo").innerHTML = Object.entries(labels).filter(([k]) => models[k]).map(([k, label]) => {
    const m = models[k];
    const pct1 = (v) => (typeof v === "number" ? `${(v * 100).toFixed(1)}%` : "n/a");
    const acc = pct1(m.test_acc);
    const f1 = pct1(m.macro_f1);
    return `<div class="model">
      <dt>${label} <span class="model-name">${escapeHtml(m.model || "")}</span></dt>
      <dd><span><strong>${acc}</strong>test accuracy</span><span><strong>${f1}</strong>macro-F1</span></dd>
    </div>`;
  }).join("");
}

let lastPayload = null;

async function submit(fd, aspect) {
  hideFormError();
  go.classList.add("busy");
  go.querySelector(".go-label").textContent = "Predicting";
  show("loading");
  dropFull.classList.add("scanning");
  const flicker = attentionFlicker(livePatchEls, 1200);
  if (window.innerWidth < 980) $("#results").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
  try {
    const data = await predict(fd);
    await flicker;
    renderResult(data, aspect);
    show("done");
  } catch (err) {
    $("#errMsg").textContent = err.message || String(err);
    show("error");
  } finally {
    dropFull.classList.remove("scanning");
    go.classList.remove("busy");
    go.querySelector(".go-label").textContent = "Predict";
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = tweet.value.trim();
  const problems = [];
  if (!text) { problems.push("a tweet"); tweet.closest(".field").classList.add("invalid"); }
  if (!imageFile) { problems.push("an image"); drop.closest(".field").classList.add("invalid"); }
  if (problems.length) {
    showFormError(`Add ${problems.join(" and ")} to get a prediction.`);
    (text ? fileIn : tweet).focus();
    return;
  }
  const aspect = aspectIn.value.trim();
  const fd = new FormData();
  fd.append("dataset", form.querySelector('input[name="dataset"]:checked').value);
  fd.append("tweet", text);
  fd.append("aspect", aspect);
  fd.append("image", imageFile, imageFile.name);
  lastPayload = { fd, aspect };
  submit(fd, aspect);
});

$("#retry").addEventListener("click", () => { if (lastPayload) submit(lastPayload.fd, lastPayload.aspect); });
