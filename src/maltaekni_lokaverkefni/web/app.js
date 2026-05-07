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
const welcomeOverlay = document.querySelector("#welcomeOverlay");
const welcomeScroll = document.querySelector("#welcomeScroll");
const welcomeScenes = Array.from(document.querySelectorAll("[data-welcome-scene]"));
const welcomeProgress = Array.from(document.querySelectorAll(".welcome-progress i"));
const startButton = document.querySelector("#startButton");
const skipWelcomeButton = document.querySelector("#skipWelcomeButton");
const tourButton = document.querySelector("#tourButton");
const tourOverlay = document.querySelector("#tourOverlay");
const tourSpotlight = document.querySelector("#tourSpotlight");
const tourCard = document.querySelector("#tourCard");
const tourCount = document.querySelector("#tourCount");
const tourTitle = document.querySelector("#tourTitle");
const tourText = document.querySelector("#tourText");
const tourNextButton = document.querySelector("#tourNextButton");
const tourSkipButton = document.querySelector("#tourSkipButton");

const introText =
  "Spurðu um gallaða vöru, netkaup, afhendingu, endurgreiðslu eða rétt til að falla frá kaupum.";

const tourSteps = [
  {
    selector: '[data-tour="settings"]',
    title: "Stillingar",
    text: "Veldu leitaraðferð og fjölda heimilda áður en þú spyrð.",
  },
  {
    selector: '[data-tour="chat"]',
    title: "Samtal",
    text: "Svör birtast hér og eru skrifuð út eins og í spjallviðmóti.",
  },
  {
    selector: '[data-tour="composer"]',
    title: "Spurning",
    text: "Settu inn hagnýta spurningu um neytendarétt og sendu hana áfram.",
  },
  {
    selector: '[data-tour="sources"]',
    title: "Heimildir",
    text: "Hér sérðu textabrotin sem svarið byggir á, með slóðum og retrieval-score.",
  },
];

let activeTourIndex = 0;
let welcomeFrame = null;

topKInput.addEventListener("input", () => {
  topKValue.textContent = topKInput.value;
});

welcomeScroll.addEventListener("scroll", () => {
  if (welcomeFrame !== null) return;
  welcomeFrame = window.requestAnimationFrame(() => {
    updateWelcomeIntro();
    welcomeFrame = null;
  });
});

welcomeOverlay.addEventListener(
  "wheel",
  (event) => {
    if (welcomeOverlay.classList.contains("hidden")) return;
    if (event.target.closest("button")) return;

    event.preventDefault();
    welcomeScroll.scrollBy({ top: event.deltaY, behavior: "auto" });
  },
  { passive: false },
);

clearButton.addEventListener("click", () => {
  messages.innerHTML = "";
  sourceList.innerHTML =
    '<p class="empty">Heimildir birtast hér eftir fyrstu spurningu.</p>';
  appendMessage("assistant", introText);
});

startButton.addEventListener("click", () => {
  closeWelcome();
  startTour();
});

skipWelcomeButton.addEventListener("click", () => {
  closeWelcome();
  questionInput.focus();
});

tourButton.addEventListener("click", () => {
  closeWelcome();
  startTour();
});

tourNextButton.addEventListener("click", () => {
  if (activeTourIndex >= tourSteps.length - 1) {
    endTour();
    return;
  }

  activeTourIndex += 1;
  renderTourStep();
});

tourSkipButton.addEventListener("click", endTour);
window.addEventListener("resize", () => {
  if (!tourOverlay.classList.contains("hidden")) renderTourStep();
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

function updateWelcomeIntro() {
  const maxScroll = Math.max(1, welcomeScroll.scrollHeight - welcomeScroll.clientHeight);
  const progress = welcomeScroll.scrollTop / maxScroll;
  const lastIndex = Math.max(1, welcomeScenes.length - 1);
  const exactIndex = progress * lastIndex;
  const activeIndex = Math.round(exactIndex);

  welcomeScenes.forEach((scene, index) => {
    const distance = Math.abs(exactIndex - index);
    const strength = Math.max(0, 1 - distance * 1.65);
    const direction = index - exactIndex;

    scene.classList.toggle("active", index === activeIndex);
    scene.style.setProperty("--scene-opacity", strength.toFixed(3));
    scene.style.setProperty("--scene-y", `${direction * 36}px`);
    scene.style.setProperty("--scene-scale", String(0.96 + strength * 0.04));
    scene.style.setProperty("--scene-blur", `${(1 - strength) * 10}px`);
  });

  welcomeProgress.forEach((item, index) => {
    item.classList.toggle("active", index === activeIndex);
  });
}

function closeWelcome() {
  welcomeOverlay.classList.add("hidden");
  document.body.classList.remove("onboarding-active");
}

function startTour() {
  activeTourIndex = 0;
  tourOverlay.classList.remove("hidden");
  tourOverlay.setAttribute("aria-hidden", "false");
  renderTourStep();
}

function endTour() {
  tourOverlay.classList.add("hidden");
  tourOverlay.setAttribute("aria-hidden", "true");
  questionInput.focus();
}

function renderTourStep() {
  const step = tourSteps[activeTourIndex];
  const target = document.querySelector(step.selector);
  if (!target) return;

  const rect = target.getBoundingClientRect();
  const padding = 8;
  const left = Math.max(10, rect.left - padding);
  const top = Math.max(10, rect.top - padding);
  const width = Math.min(window.innerWidth - left - 10, rect.width + padding * 2);
  const height = Math.min(window.innerHeight - top - 10, rect.height + padding * 2);

  tourSpotlight.style.left = `${left}px`;
  tourSpotlight.style.top = `${top}px`;
  tourSpotlight.style.width = `${width}px`;
  tourSpotlight.style.height = `${height}px`;

  tourCount.textContent = `${activeTourIndex + 1} / ${tourSteps.length}`;
  tourTitle.textContent = step.title;
  tourText.textContent = step.text;
  tourNextButton.textContent =
    activeTourIndex === tourSteps.length - 1 ? "Ljúka" : "Næst";

  positionTourCard({ left, top, width, height });
}

function positionTourCard(rect) {
  const gap = 14;
  const cardWidth = Math.min(330, window.innerWidth - 28);
  const cardHeight = 190;
  let left = rect.left + rect.width + gap;
  let top = rect.top;

  if (left + cardWidth > window.innerWidth - 14) {
    left = rect.left;
    top = rect.top + rect.height + gap;
  }

  if (top + cardHeight > window.innerHeight - 14) {
    top = Math.max(14, rect.top - cardHeight - gap);
  }

  left = Math.max(14, Math.min(left, window.innerWidth - cardWidth - 14));
  top = Math.max(14, top);
  tourCard.style.left = `${left}px`;
  tourCard.style.top = `${top}px`;
}

window.addEventListener("load", () => {
  if (window.lucide) window.lucide.createIcons();
  document.body.classList.add("onboarding-active");
  updateWelcomeIntro();
  checkStatus();
});
