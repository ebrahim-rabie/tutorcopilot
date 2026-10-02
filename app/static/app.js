"use strict";

const $ = (id) => document.getElementById(id);

const messagesEl = $("messages");
const inputEl = $("input");
const sendBtn = $("send");
const stopBtn = $("stop");
const bannerEl = $("banner");
const micBtn = $("mic");
const ttsBtn = $("tts");

let conversationId = null;
let busy = false;
let abortCtrl = null;

/* ---------------- helpers ---------------- */

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}

function scrollDown() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function showBanner(msg, isError) {
  bannerEl.textContent = msg;
  bannerEl.classList.toggle("error", !!isError);
  bannerEl.classList.remove("hidden");
}

function hideBanner() {
  bannerEl.classList.add("hidden");
}

function fmtJson(obj) {
  try {
    return JSON.stringify(obj, null, 2);
  } catch {
    return String(obj);
  }
}

function truncate(s, n) {
  return s.length > n ? s.slice(0, n) + "\n…" : s;
}

/* ---------------- sidebar ---------------- */

async function loadHealth() {
  const badge = $("health");
  try {
    const h = await (await fetch("/health")).json();
    health = h;
    updateTtsBtn();
    const configured = !!h.llm_configured;
    badge.textContent = `${h.index_chunks} chunks · LLM ${configured ? "✓" : "✗"}`;
    badge.className = "badge " + (configured ? "ok" : "bad");
    if (!configured) {
      showBanner("LLM not configured — put TC_LLM_API_KEY=gsk_… in .env and restart the server.");
    } else {
      hideBanner();
    }
  } catch {
    badge.textContent = "server offline";
    badge.className = "badge bad";
  }
}

async function loadWeakTopics() {
  const target = $("weak-topic");
  try {
    const w = await (await fetch("/agent/memory/weak-topics")).json();
    target.textContent = w.weakest_topic ? `⚠ ${w.weakest_topic}` : "none yet";
    target.classList.toggle("muted", !w.weakest_topic);
  } catch {
    target.textContent = "—";
  }
}

function updateConvLabel() {
  $("conv-id").textContent = conversationId || "new";
}

async function saveAndNew() {
  if (busy) return;
  if (conversationId) {
    try {
      await fetch(`/agent/conversations/${conversationId}`, { method: "DELETE" });
    } catch {
      /* server unreachable — conversation stays in memory only */
    }
  }
  conversationId = null;
  updateConvLabel();
  messagesEl.innerHTML = "";
  const w = el("div", "welcome", "");
  w.id = "welcome";
  w.append(
    el("h2", null, "Ask your tutor"),
    el("p", null, "Questions about RAG, agents, chunking, evaluation, fine-tuning…"),
    el("p", "muted", "While the agent works you'll see its thoughts and tool calls live, step by step.")
  );
  messagesEl.appendChild(w);
  loadWeakTopics();
}

/* ---------------- message rendering ---------------- */

function removeWelcome() {
  const w = $("welcome");
  if (w) w.remove();
}

function addUserMsg(text) {
  messagesEl.appendChild(el("div", "msg user", text));
  scrollDown();
}

function createAssistantMsg() {
  const wrap = el("div", "msg assistant");
  const steps = el("div", "steps");
  const thinking = el("div", "thinking", "thinking ");
  const answer = el("div", "answer");
  const meta = el("div", "meta hidden");
  wrap.append(steps, thinking, answer, meta);
  messagesEl.appendChild(wrap);
  scrollDown();
  return { wrap, steps, thinking, answer, meta, answered: false };
}

function addThought(bubble, iteration, text) {
  const card = el("details", "step thought");
  card.append(el("summary", null, `💭 Thought · step ${iteration}`));
  const body = el("div", "body", text || "(empty)");
  card.appendChild(body);
  bubble.steps.appendChild(card);
  scrollDown();
}

function addToolCall(bubble, tool, args) {
  const card = el("details", "step tool");
  card.append(el("summary", null, `🔧 ${tool}`));
  card.appendChild(el("div", "args", fmtJson(args)));
  card.dataset.tool = tool;
  card.dataset.done = "0";
  bubble.steps.appendChild(card);
  scrollDown();
  return card;
}

function addToolResult(bubble, pending, tool, result) {
  let card = pending.find((c) => c.dataset.tool === tool && c.dataset.done === "0");
  if (!card) {
    card = addToolCall(bubble, tool, {});
  }
  card.dataset.done = "1";
  const ok = !!(result && result.success);
  const chips = el("div", "chips");
  chips.append(el("span", "chip " + (ok ? "ok" : "fail"), ok ? "ok" : "fail"));
  if (result && result.latency_ms != null) {
    chips.append(el("span", "chip time", `${result.latency_ms} ms`));
  }
  const details = el("details");
  details.append(el("summary", null, ok ? "result" : `error: ${result && result.error ? result.error : "failed"}`));
  const raw = result ? result.data : null;
  const body = typeof raw === "string" ? raw : fmtJson(raw);
  const pre = el("div", "body", truncate(body || "(no data)", 400));
  details.appendChild(pre);
  card.append(chips, details);
  scrollDown();
}

function addWarn(bubble, text) {
  const card = el("div", "step warn");
  card.append(el("summary", null, `⚠ ${text}`));
  bubble.steps.appendChild(card);
  scrollDown();
}

function finishBubble(bubble, metaText) {
  bubble.thinking.classList.add("hidden");
  if (!bubble.answered) {
    bubble.answer.textContent = "(stream ended without an answer)";
    bubble.answered = true;
  }
  if (metaText) {
    bubble.meta.textContent = metaText;
    bubble.meta.classList.remove("hidden");
  }
  scrollDown();
}

/* ---------------- SSE ---------------- */

function handleSseBlock(block, bubble, pending) {
  let type = null;
  let data = {};
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) {
      type = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      try {
        data = JSON.parse(line.slice(5).trim());
      } catch {
        /* keep previous data */
      }
    }
  }
  if (!type) return;

  switch (type) {
    case "step_start":
      bubble.thinking.textContent = `step ${data.iteration} — planning `;
      break;

    case "thought":
      addThought(bubble, data.iteration, data.thought);
      break;

    case "tool_call":
      pending.push(addToolCall(bubble, data.tool, data.args));
      break;

    case "tool_result":
      addToolResult(bubble, pending, data.tool, data.result);
      break;

    case "nudge":
      addWarn(bubble, data.message || "nudge");
      break;

    case "max_iterations":
      addWarn(bubble, data.message || "iteration limit reached");
      break;

    case "final_answer":
      bubble.thinking.classList.add("hidden");
      bubble.answer.textContent = data.answer || "";
      bubble.answered = true;
      speakAnswer(data.answer || "");
      scrollDown();
      break;

    case "done": {
      if (data.conversation_id) {
        conversationId = data.conversation_id;
        updateConvLabel();
      }
      const tools = (data.tools_called || []).join(", ") || "none";
      const tok = data.tokens_used || {};
      const parts = [`tools: ${tools}`, `tokens ${tok.prompt || 0}/${tok.completion || 0}`];
      if (data.trace_id) parts.push(`trace ${data.trace_id}`);
      finishBubble(bubble, parts.join(" · "));
      break;
    }

    case "error":
      bubble.wrap.classList.add("error");
      bubble.thinking.classList.add("hidden");
      bubble.answer.textContent = "Error: " + (data.error || data.message || "unknown");
      bubble.answered = true;
      break;

    case "close":
      finishBubble(bubble, bubble.meta.textContent || "");
      break;

    default:
      break;
  }
}

/* ---------------- sending ---------------- */

function setBusy(b) {
  busy = b;
  sendBtn.disabled = b;
  stopBtn.classList.toggle("hidden", !b);
}

async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text || busy) return;

  hideBanner();
  removeWelcome();
  stopSpeaking();
  addUserMsg(text);
  inputEl.value = "";
  const bubble = createAssistantMsg();
  const pending = [];
  setBusy(true);
  abortCtrl = new AbortController();

  try {
    const res = await fetch("/agent/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: text, conversation_id: conversationId }),
      signal: abortCtrl.signal,
    });

    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try {
        const j = await res.json();
        if (typeof j.detail === "string") detail = j.detail;
      } catch {
        /* non-JSON error body */
      }
      throw new Error(detail);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        handleSseBlock(block, bubble, pending);
      }
    }
    finishBubble(bubble, bubble.meta.textContent || "");
  } catch (err) {
    if (err.name === "AbortError") {
      bubble.thinking.classList.add("hidden");
      if (!bubble.answered) {
        bubble.answer.textContent = "(stopped)";
        bubble.answered = true;
      }
    } else {
      bubble.thinking.classList.add("hidden");
      bubble.wrap.classList.add("error");
      bubble.answer.textContent = err.message;
      bubble.answered = true;
      if (/LLM not configured/i.test(err.message)) {
        showBanner(err.message, true);
      } else if (!/abort/i.test(err.message)) {
        showBanner(err.message, true);
      }
    }
  } finally {
    abortCtrl = null;
    setBusy(false);
    scrollDown();
    loadWeakTopics();
  }
}

/* ---------------- voice: STT + TTS ---------------- */

const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
let health = {};
let ttsMode = localStorage.getItem("tc_tts") || "off";
let recognition = null;
let recorder = null;
let recordingTimer = null;
let listening = false;
let srFallbackTried = false;

function updateTtsBtn() {
  if (ttsMode === "server" && health.tts_provider !== "edge") ttsMode = "off";
  const icons = { off: "\u{1F507}", browser: "\u{1F50A}", server: "\u{1F508}" };
  const labels = {
    off: "Answer voice: off",
    browser: "Answer voice: browser speech",
    server: "Answer voice: server (edge-tts)",
  };
  ttsBtn.textContent = icons[ttsMode] || icons.off;
  ttsBtn.title = labels[ttsMode] || labels.off;
}

function cycleTts() {
  const modes = ["off", "browser"];
  if (health.tts_provider === "edge") modes.push("server");
  const idx = modes.indexOf(ttsMode);
  ttsMode = modes[(idx + 1) % modes.length];
  if (ttsMode === "off") stopSpeaking();
  localStorage.setItem("tc_tts", ttsMode);
  updateTtsBtn();
}

function speakable(text) {
  return text
    .replace(/\[\d+(?:,\s*\d+)*\]/g, " ")
    .replace(/[*_#`>|]/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 4000);
}

function stopSpeaking() {
  if ("speechSynthesis" in window) window.speechSynthesis.cancel();
}

function speakBrowser(text) {
  if (!("speechSynthesis" in window) || !text) return;
  stopSpeaking();
  const u = new SpeechSynthesisUtterance(text);
  u.lang = "en-US";
  window.speechSynthesis.speak(u);
}

function speakAnswer(raw) {
  if (ttsMode === "off") return;
  const text = speakable(raw || "");
  if (!text) return;
  if (ttsMode === "server") {
    fetch("/agent/speak", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    })
      .then((res) => {
        if (!res.ok) throw new Error(`TTS HTTP ${res.status}`);
        return res.blob();
      })
      .then((blob) => {
        const audio = new Audio(URL.createObjectURL(blob));
        audio.play().catch(() => {});
      })
      .catch(() => speakBrowser(text));
  } else {
    speakBrowser(text);
  }
}

function setListening(on) {
  listening = on;
  micBtn.classList.toggle("listening", on);
}

function toggleMic() {
  if (listening) {
    if (recorder && recorder.state === "recording") recorder.stop();
    else if (recognition) recognition.stop();
    return;
  }
  stopSpeaking();
  hideBanner();
  if (SR) startBrowserRecognition();
  else startRecorder();
}

function startBrowserRecognition() {
  try {
    recognition = new SR();
  } catch {
    startRecorder();
    return;
  }
  recognition.lang = "en-US";
  recognition.interimResults = true;
  recognition.continuous = false;
  recognition.onresult = (e) => {
    let finalText = "";
    let interim = "";
    for (const res of e.results) {
      if (res.isFinal) finalText += res[0].transcript;
      else interim += res[0].transcript;
    }
    if (finalText) {
      inputEl.value = finalText.trim();
      if (!busy) sendMessage();
    } else {
      inputEl.value = interim;
    }
  };
  recognition.onerror = (e) => {
    setListening(false);
    if (e.error === "not-allowed" || e.error === "service-not-allowed") {
      showBanner("Microphone blocked — allow microphone access or type instead.", true);
    } else if (e.error === "no-speech") {
      showBanner("Didn't hear anything — try again.", false);
    } else if (e.error === "network" && !srFallbackTried) {
      srFallbackTried = true;
      startRecorder();
    }
  };
  recognition.onend = () => setListening(false);
  try {
    recognition.start();
    setListening(true);
  } catch {
    setListening(false);
    startRecorder();
  }
}

function startRecorder() {
  if (!navigator.mediaDevices || !window.MediaRecorder) {
    showBanner("No speech input in this browser — type instead.", true);
    return;
  }
  navigator.mediaDevices
    .getUserMedia({ audio: true })
    .then((stream) => {
      const chunks = [];
      const rec = new MediaRecorder(stream);
      rec.ondataavailable = (e) => {
        if (e.data && e.data.size) chunks.push(e.data);
      };
      rec.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        clearTimeout(recordingTimer);
        setListening(false);
        uploadRecording(rec.mimeType, chunks);
      };
      recorder = rec;
      rec.start();
      setListening(true);
      showBanner("Recording — press the mic to stop (max 30s).", false);
      recordingTimer = setTimeout(() => {
        if (rec.state === "recording") rec.stop();
      }, 30000);
    })
    .catch(() => showBanner("Microphone unavailable — permission denied or no device.", true));
}

async function uploadRecording(mime, chunks) {
  micBtn.textContent = "⏳";
  try {
    const type = mime && mime.startsWith("audio/") ? mime : "audio/webm";
    const fd = new FormData();
    fd.append("audio", new Blob(chunks, { type }), "audio.webm");
    const res = await fetch("/agent/transcribe", { method: "POST", body: fd });
    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try {
        const j = await res.json();
        if (typeof j.detail === "string") detail = j.detail;
      } catch {
        /* non-JSON error body */
      }
      throw new Error(detail);
    }
    const { text } = await res.json();
    if (text) {
      inputEl.value = text;
      if (!busy) sendMessage();
    } else {
      showBanner("No speech detected — try again.", false);
    }
  } catch (err) {
    showBanner("Transcription failed: " + err.message, true);
  } finally {
    micBtn.textContent = "\u{1F3A4}";
  }
}

/* ---------------- wiring ---------------- */

$("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  sendMessage();
});

inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    sendMessage();
  }
});

stopBtn.addEventListener("click", () => {
  stopSpeaking();
  if (abortCtrl) abortCtrl.abort();
});

$("btn-new").addEventListener("click", saveAndNew);
$("refresh-weak").addEventListener("click", loadWeakTopics);
micBtn.addEventListener("click", toggleMic);
ttsBtn.addEventListener("click", cycleTts);

updateTtsBtn();
loadHealth();
loadWeakTopics();
setInterval(loadHealth, 15000);
