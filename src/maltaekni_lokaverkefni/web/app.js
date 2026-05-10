// Main chat UI for Réttarvísir. This file owns the onboarding screens, guided
// tour, question submission, typed answer animation, and source rendering.
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
const methodologyButton = document.querySelector("#methodologyButton");
const methodologyOverlay = document.querySelector("#methodologyOverlay");
const methodologyCloseButton = document.querySelector("#methodologyCloseButton");
const accessOverlay = document.querySelector("#accessOverlay");
const accessForm = document.querySelector("#accessForm");
const accessTokenInput = document.querySelector("#accessToken");
const accessError = document.querySelector("#accessError");
const accessCancelButton = document.querySelector("#accessCancelButton");
const pageParams = new URLSearchParams(window.location.search);
const skipWelcome = pageParams.get("skipWelcome") === "1";
const forceWelcome = pageParams.get("welcome") === "1";
const openMethodologyOnLoad = pageParams.get("methodology") === "1";
const accessTokenStorageKey = "rettarvisir_access_token";

const introText =
  "Spurðu um neytendarétt. Ég svara með heimildum og sýni textabrotin sem styðja niðurstöðuna.";

const tourSteps = [
  {
    selector: '[data-tour="settings"]',
    title: "Stillingar",
    text: "Veldu leitaraðferð og hversu mörg heimildabrot Réttarvísir á að nota.",
  },
  {
    selector: '[data-tour="chat"]',
    title: "Samtal",
    text: "Hér birtist svarið, skrifað út í rólegu spjallflæði.",
  },
  {
    selector: '[data-tour="composer"]',
    title: "Spurning",
    text: "Spurðu eins og þú myndir spyrja lögfræðing eða ráðgjafa.",
  },
  {
    selector: '[data-tour="sources"]',
    title: "Heimildir",
    text: "Hér sérðu heimildabrotin sem styðja svarið, með slóðum og vægi.",
  },
];

let activeTourIndex = 0;
let welcomeFrame = null;
let accessRequired = false;

topKInput.addEventListener("input", () => {
  topKValue.textContent = topKInput.value;
});

welcomeScroll.addEventListener("scroll", () => {
  // The intro animation updates several CSS variables, so throttle it to one
  // update per animation frame while the user scrolls.
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
    welcomeScroll.scrollBy({ top: event.deltaY * 1.4, behavior: "auto" });
  },
  { passive: false },
);

clearButton.addEventListener("click", () => {
  messages.innerHTML = "";
  sourceList.innerHTML =
    '<p class="empty">Heimildabrot birtast hér eftir fyrstu spurningu.</p>';
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

methodologyButton.addEventListener("click", () => {
  closeWelcome();
  openMethodology();
});

methodologyCloseButton.addEventListener("click", closeMethodology);
methodologyOverlay.addEventListener("click", (event) => {
  if (event.target === methodologyOverlay) closeMethodology();
});

accessForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const token = accessTokenInput.value.trim();
  if (!token) {
    accessError.textContent = "Aðgangslykill vantar.";
    return;
  }

  localStorage.setItem(accessTokenStorageKey, token);
  closeAccessPrompt();
  questionInput.focus();
});

accessCancelButton.addEventListener("click", closeAccessPrompt);

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
  if (accessRequired && !getAccessToken()) {
    openAccessPrompt();
    return;
  }

  appendMessage("user", question);
  questionInput.value = "";
  sendButton.disabled = true;
  const pending = appendMessage("assistant", "Sæki heimildir...");

  try {
    // /api/ask returns an AnswerResult dictionary: answer text, sources,
    // confidence metadata, retrieval method, and optional token/cost usage.
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: requestHeaders(),
      body: JSON.stringify({
        question,
        method: methodInput.value,
        top_k: Number(topKInput.value),
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      if (response.status === 401) {
        localStorage.removeItem(accessTokenStorageKey);
        openAccessPrompt("Aðgangslykillinn var ekki réttur.");
      }
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
  // The status badge only checks whether processed chunks exist on the server.
  try {
    const response = await fetch("/api/status");
    const status = await response.json();
    statusEl.classList.toggle("ready", status.ready);
    statusEl.classList.toggle("error", !status.ready);
    accessRequired = Boolean(status.access_required);
    statusEl.querySelector("span:last-child").textContent = status.ready
      ? (accessRequired ? "Tilbúið · læst" : "Tilbúið")
      : "Vantar gögn";
  } catch {
    statusEl.classList.add("error");
    statusEl.querySelector("span:last-child").textContent = "Nær ekki sambandi";
  }
}

function requestHeaders() {
  const headers = { "Content-Type": "application/json" };
  const accessToken = getAccessToken();
  if (accessToken) headers["X-App-Access-Token"] = accessToken;
  return headers;
}

function getAccessToken() {
  return localStorage.getItem(accessTokenStorageKey) || "";
}

function openAccessPrompt(message = "") {
  closeWelcome();
  accessError.textContent = message;
  accessOverlay.classList.remove("hidden");
  accessOverlay.setAttribute("aria-hidden", "false");
  accessTokenInput.value = "";
  window.setTimeout(() => accessTokenInput.focus(), 0);
}

function closeAccessPrompt() {
  accessOverlay.classList.add("hidden");
  accessOverlay.setAttribute("aria-hidden", "true");
  accessError.textContent = "";
}

function appendMessage(role, text) {
  // Render one chat bubble and return the wrapper so callers can update it
  // later, for example replacing "Sæki heimildir..." with the real answer.
  const section = document.createElement("section");
  section.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "Þ" : "[§]";

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
  // A small typing effect makes LLM responses easier to follow in the demo UI.
  const paragraph = document.createElement("p");
  target.append(paragraph);

  for (const char of text) {
    paragraph.textContent += char;
    messages.scrollTop = messages.scrollHeight;
    await wait(11);
  }
}

function appendMeta(target, result) {
  // Confidence, citation coverage, and token usage are shown below the answer
  // but kept separate from the answer text itself.
  const meta = document.createElement("div");
  meta.className = "meta";
  const usageText = formatUsage(result.usage || {});
  const coverageText = formatSourceCoverage(result.source_coverage || {});
  meta.innerHTML = `
    <span>Traust: ${escapeHtml(result.confidence)} · Aðferð: ${escapeHtml(result.method)}</span>
    <span>${escapeHtml(result.confidence_reason || "")}</span>
    ${coverageText ? `<span>${escapeHtml(coverageText)}</span>` : ""}
    ${usageText ? `<span>${escapeHtml(usageText)}</span>` : ""}
  `;
  target.append(meta);
}

function renderSources(sources) {
  // Source cards mirror the backend citation ids so the answer can refer to
  // [1], [2], etc. without hiding the underlying legal text.
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
      <p class="source-reason">${escapeHtml(source.reason || "")}</p>
      <a href="${escapeAttribute(source.url)}" target="_blank" rel="noreferrer">${escapeHtml(source.url)}</a>
      <div class="score">Vægi: ${formatScore(source.score)} · ${escapeHtml(source.retrieval_method || "")}</div>
    `;
    sourceList.append(card);
  }
}

function formatScore(score) {
  if (score === null || score === undefined) return "n/a";
  return Number(score).toFixed(4);
}

function formatUsage(usage) {
  // Token fields are optional because extractive fallback answers do not call
  // an LLM and some providers may omit usage metadata.
  const totalTokens = Number(usage.total_tokens);
  if (!Number.isFinite(totalTokens) || totalTokens <= 0) return "";

  const inputTokens = Number(usage.prompt_tokens);
  const outputTokens = Number(usage.output_tokens);
  const parts = [`Tokenar: ${totalTokens.toLocaleString("is-IS")}`];
  if (Number.isFinite(inputTokens)) parts.push(`inn ${inputTokens.toLocaleString("is-IS")}`);
  if (Number.isFinite(outputTokens)) parts.push(`út ${outputTokens.toLocaleString("is-IS")}`);

  const estimatedCost = Number(usage.estimated_cost_usd);
  if (Number.isFinite(estimatedCost) && estimatedCost > 0) {
    parts.push(`~$${estimatedCost.toFixed(6)}`);
  }

  return parts.join(" · ");
}

function formatSourceCoverage(sourceCoverage) {
  const cited = Number(sourceCoverage.cited_source_count);
  const total = Number(sourceCoverage.source_count);
  if (!Number.isFinite(cited) || !Number.isFinite(total) || total <= 0) return "";

  return `Heimildanotkun: ${cited}/${total} heimildir vísaðar í svarinu`;
}

function escapeHtml(value) {
  return String(value)
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
  // Convert scroll position into one active scene plus continuous opacity,
  // vertical offset, scale, and blur values for neighboring scenes.
  const maxScroll = Math.max(1, welcomeScroll.scrollHeight - welcomeScroll.clientHeight);
  const progress = welcomeScroll.scrollTop / maxScroll;
  const lastIndex = Math.max(1, welcomeScenes.length - 1);
  const exactIndex = progress * lastIndex;
  const activeIndex = Math.round(exactIndex);

  welcomeScenes.forEach((scene, index) => {
    const distance = Math.abs(exactIndex - index);
    const strength = welcomeSceneStrength(distance);
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

function welcomeSceneStrength(distance) {
  const holdDistance = 0.22;
  const fadeDistance = 0.62;

  if (distance <= holdDistance) return 1;
  if (distance >= fadeDistance) return 0;

  const fadeProgress = (distance - holdDistance) / (fadeDistance - holdDistance);
  return 1 - smoothStep(fadeProgress);
}

function smoothStep(value) {
  return value * value * (3 - 2 * value);
}

function closeWelcome() {
  welcomeOverlay.classList.add("hidden");
  document.body.classList.remove("onboarding-active");
  document.body.classList.add("app-entered");
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

function closeMethodology() {
  methodologyOverlay.classList.add("hidden");
  methodologyOverlay.setAttribute("aria-hidden", "true");
  questionInput.focus();
}

function openMethodology() {
  methodologyOverlay.classList.remove("hidden");
  methodologyOverlay.setAttribute("aria-hidden", "false");
}

function renderTourStep() {
  // Position the spotlight around the target control and move the explanatory
  // card to a visible side of the highlighted element.
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
  // Query parameters let evaluation links jump straight into the app or open
  // the methodology modal without requiring manual navigation.
  if (window.lucide) window.lucide.createIcons();
  if (openMethodologyOnLoad) {
    closeWelcome();
    openMethodology();
  } else if (skipWelcome && !forceWelcome) {
    closeWelcome();
  } else {
    document.body.classList.add("onboarding-active");
    updateWelcomeIntro();
  }
  checkStatus();
});
