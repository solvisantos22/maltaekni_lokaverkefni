const messages = document.querySelector("#messages");
const sourceList = document.querySelector("#sourceList");
const form = document.querySelector("#askForm");
const questionInput = document.querySelector("#question");
const sendButton = document.querySelector("#sendButton");
const methodInput = document.querySelector("#method");
const topKInput = document.querySelector("#topK");
const topKValue = document.querySelector("#topKValue");
const statusEl = document.querySelector("#status");
const clearButton = document.querySelector("#clearButton");

topKInput.addEventListener("input", () => {
  topKValue.textContent = topKInput.value;
});

clearButton.addEventListener("click", () => {
  messages.innerHTML = "";
  sourceList.innerHTML = '<p class="empty">Heimildir birtast hér eftir fyrstu spurningu.</p>';
  appendMessage("assistant", "Spurðu um gallaða vöru, netkaup, afhendingu, endurgreiðslu eða rétt til að falla frá kaupum.");
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  appendMessage("user", question);
  questionInput.value = "";
  sendButton.disabled = true;
  const pending = appendMessage("assistant", "Sæki heimildir...");

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        method: methodInput.value,
        top_k: Number(topKInput.value),
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Villa kom upp.");
    }

    const result = await response.json();
    pending.querySelector(".bubble").innerHTML = "";
    await typeAnswer(pending.querySelector(".bubble"), result.answer);
    appendMeta(pending.querySelector(".bubble"), result);
    renderSources(result.sources);
  } catch (error) {
    pending.querySelector(".bubble").textContent = error.message;
  } finally {
    sendButton.disabled = false;
    questionInput.focus();
  }
});

async function checkStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();
    statusEl.classList.toggle("ready", status.ready);
    statusEl.classList.toggle("error", !status.ready);
    statusEl.querySelector("span:last-child").textContent = status.ready
      ? "Tilbúið"
      : "Vantar gögn";
  } catch {
    statusEl.classList.add("error");
    statusEl.querySelector("span:last-child").textContent = "Nær ekki sambandi";
  }
}

function appendMessage(role, text) {
  const section = document.createElement("section");
  section.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "Þ" : "§";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  const paragraph = document.createElement("p");
  paragraph.textContent = text;
  bubble.append(paragraph);

  section.append(avatar, bubble);
  messages.append(section);
  messages.scrollTop = messages.scrollHeight;
  return section;
}

async function typeAnswer(target, text) {
  const paragraph = document.createElement("p");
  target.append(paragraph);

  for (const char of text) {
    paragraph.textContent += char;
    messages.scrollTop = messages.scrollHeight;
    await wait(11);
  }
}

function appendMeta(target, result) {
  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = `Traust: ${result.confidence} · Aðferð: ${result.method}`;
  target.append(meta);
}

function renderSources(sources) {
  if (!sources.length) {
    sourceList.innerHTML = '<p class="empty">Engar heimildir fundust.</p>';
    return;
  }

  sourceList.innerHTML = "";
  for (const source of sources) {
    const card = document.createElement("article");
    card.className = "source-card";
    card.innerHTML = `
      <span class="badge">[${escapeHtml(String(source.citation_id))}]</span>
      <h3>${escapeHtml(source.title)} - ${escapeHtml(source.section)}</h3>
      <p>${escapeHtml(source.text)}</p>
      <a href="${escapeAttribute(source.url)}" target="_blank" rel="noreferrer">${escapeHtml(source.url)}</a>
      <div class="score">Score: ${formatScore(source.score)} · ${escapeHtml(source.retrieval_method || "")}</div>
    `;
    sourceList.append(card);
  }
}

function formatScore(score) {
  if (score === null || score === undefined) return "n/a";
  return Number(score).toFixed(4);
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function escapeAttribute(value) {
  return escapeHtml(value || "#");
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

window.addEventListener("load", () => {
  if (window.lucide) window.lucide.createIcons();
  checkStatus();
});
