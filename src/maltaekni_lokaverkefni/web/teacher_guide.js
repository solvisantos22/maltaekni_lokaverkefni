// Floating checklist used during project/teacher review. The guide is injected
// into every web page that includes this script, and its UI state is persisted
// locally so reviewers can move between pages without losing progress.
(function () {
  // localStorage keys are namespaced so they do not collide with app state.
  const collapsedKey = "rettarvisirTeacherGuideCollapsed";
  const checkedKey = "rettarvisirTeacherGuideChecked";
  const positionKey = "rettarvisirTeacherGuidePosition";
  const sessionStartedKey = "rettarvisirTeacherGuideSessionStarted";
  const defaultPosition = { left: 24, top: 96 };
  const items = [
    {
      id: "chat",
      label: "Prófa spjallið",
      href: "/?skipWelcome=1#chatPanel",
    },
    {
      id: "sources",
      label: "Lesa heimildir og tilvísanir",
      href: "/?skipWelcome=1#sourcesPanel",
    },
    {
      id: "methods",
      label: "Prófa aðra leitaraðferð",
      href: "/?skipWelcome=1#settingsPanel",
    },
    {
      id: "methodology",
      label: "Lesa Aðferð",
      href: "/?skipWelcome=1&methodology=1",
    },
    {
      id: "dashboard",
      label: "Skoða mælaborð mats",
      href: "/evaluation/dashboard",
    },
    {
      id: "review",
      label: "Skoða mannlegt mat",
      href: "/evaluation",
    },
  ];

  const checked = readJson(checkedKey, {});
  const guide = document.createElement("aside");
  guide.className = "teacher-guide";
  guide.setAttribute("aria-label", "Leið fyrir yfirferð kennara");
  guide.innerHTML = `
    <div class="teacher-guide-head" data-drag-handle>
      <div class="teacher-guide-title">
        <strong>Yfirferð</strong>
        <span>Dragðu gluggann og hakaðu við það sem er skoðað</span>
      </div>
      <button class="teacher-guide-toggle" type="button" aria-label="Fela leiðbeiningar">-</button>
    </div>
    <ul class="teacher-guide-list">
      ${items
        .map(
          (item) => `
            <li>
              <label>
                <input type="checkbox" data-guide-check="${item.id}" ${checked[item.id] ? "checked" : ""} />
                <span>${item.label}</span>
              </label>
              <a href="${item.href}" aria-label="${item.label}">Opna</a>
            </li>
          `,
        )
        .join("")}
    </ul>
  `;

  const bubble = document.createElement("button");
  bubble.className = "teacher-guide-bubble";
  bubble.type = "button";
  bubble.innerHTML = "<span>§</span> Yfirferð";
  bubble.setAttribute("aria-label", "Opna leiðbeiningar");

  const toggleButton = guide.querySelector(".teacher-guide-toggle");
  const dragHandle = guide.querySelector("[data-drag-handle]");
  let dragState = null;
  let suppressBubbleClick = false;

  function readJson(key, fallback) {
    // Invalid stored JSON should not break the page; fall back silently.
    try {
      return JSON.parse(localStorage.getItem(key) || "null") || fallback;
    } catch {
      return fallback;
    }
  }

  function writeJson(key, value) {
    localStorage.setItem(key, JSON.stringify(value));
  }

  function setCollapsed(collapsed, options = {}) {
    // Preserve the current top-right corner when switching between the large
    // checklist and compact bubble, so collapsing does not visually jump.
    const previous = collapsed ? guide : bubble;
    const previousRect = previous.getBoundingClientRect();
    guide.hidden = collapsed;
    bubble.hidden = !collapsed;
    localStorage.setItem(collapsedKey, collapsed ? "1" : "0");
    if (options.useDefaultPosition) {
      applyDefaultPosition(collapsed ? bubble : guide);
      return;
    }
    if (collapsed && previousRect.width > 0) {
      const bubbleWidth = bubble.offsetWidth || 112;
      const next = clampPosition(previousRect.right - bubbleWidth, previousRect.top, bubble);
      writeJson(positionKey, next);
      applyStoredPosition(bubble);
      return;
    }
    if (!collapsed && previousRect.width > 0) {
      const guideWidth = guide.offsetWidth || 330;
      const next = clampPosition(previousRect.right - guideWidth, previousRect.top, guide);
      writeJson(positionKey, next);
      applyStoredPosition(guide);
      return;
    }
    applyStoredPosition(collapsed ? bubble : guide);
  }

  function applyDefaultPosition(element) {
    const next = clampPosition(defaultPosition.left, defaultPosition.top, element);
    element.style.left = `${next.left}px`;
    element.style.top = `${next.top}px`;
    element.style.right = "auto";
    element.style.bottom = "auto";
  }

  function applyStoredPosition(element) {
    const position = readJson(positionKey, null);
    if (position && Number.isFinite(position.left) && Number.isFinite(position.top)) {
      element.style.left = `${position.left}px`;
      element.style.top = `${position.top}px`;
      element.style.right = "auto";
      element.style.bottom = "auto";
      return;
    }

    applyDefaultPosition(element);
  }

  function clampPosition(left, top, element) {
    // Keep the floating guide visible after dragging or viewport resize.
    const margin = 8;
    const width = element.offsetWidth || 280;
    const height = element.offsetHeight || 220;
    return {
      left: Math.max(margin, Math.min(left, window.innerWidth - width - margin)),
      top: Math.max(margin, Math.min(top, window.innerHeight - height - margin)),
    };
  }

  function beginDrag(event, element, options = {}) {
    // Dragging the full guide should ignore controls so links and checkboxes
    // still behave normally. The compact bubble is itself draggable.
    if (!options.allowControls && event.target.closest("button, a, input, label")) return;
    const rect = element.getBoundingClientRect();
    dragState = {
      pointerId: event.pointerId,
      element,
      moved: false,
      offsetX: event.clientX - rect.left,
      offsetY: event.clientY - rect.top,
    };
    element.setPointerCapture(event.pointerId);
    element.classList.add("dragging");
  }

  function moveDrag(event) {
    if (!dragState || event.pointerId !== dragState.pointerId) return;
    const next = clampPosition(
      event.clientX - dragState.offsetX,
      event.clientY - dragState.offsetY,
      dragState.element,
    );
    dragState.moved = true;
    dragState.element.style.left = `${next.left}px`;
    dragState.element.style.top = `${next.top}px`;
    dragState.element.style.right = "auto";
    dragState.element.style.bottom = "auto";
  }

  function endDrag(event) {
    if (!dragState || event.pointerId !== dragState.pointerId) return;
    const element = dragState.element;
    const rect = element.getBoundingClientRect();
    const next = clampPosition(rect.left, rect.top, element);
    writeJson(positionKey, next);
    if (element === bubble && dragState.moved) {
      // A pointerup after dragging the bubble would otherwise also trigger the
      // click handler and reopen the full checklist.
      suppressBubbleClick = true;
      window.setTimeout(() => {
        suppressBubbleClick = false;
      }, 0);
    }
    element.releasePointerCapture(event.pointerId);
    element.classList.remove("dragging");
    dragState = null;
  }

  toggleButton.addEventListener("click", () => setCollapsed(true));
  bubble.addEventListener("click", () => {
    if (suppressBubbleClick) return;
    setCollapsed(false);
  });
  dragHandle.addEventListener("pointerdown", (event) => beginDrag(event, guide));
  bubble.addEventListener("pointerdown", (event) => beginDrag(event, bubble, { allowControls: true }));
  guide.addEventListener("pointermove", moveDrag);
  guide.addEventListener("pointerup", endDrag);
  guide.addEventListener("pointercancel", endDrag);
  bubble.addEventListener("pointermove", moveDrag);
  bubble.addEventListener("pointerup", endDrag);
  bubble.addEventListener("pointercancel", endDrag);
  window.addEventListener("resize", () => {
    const active = guide.hidden ? bubble : guide;
    const rect = active.getBoundingClientRect();
    writeJson(positionKey, clampPosition(rect.left, rect.top, active));
    applyStoredPosition(active);
  });

  guide.addEventListener("change", (event) => {
    const checkbox = event.target.closest("[data-guide-check]");
    if (!checkbox) return;
    checked[checkbox.dataset.guideCheck] = checkbox.checked;
    writeJson(checkedKey, checked);
  });

  document.addEventListener("DOMContentLoaded", () => {
    document.body.append(guide, bubble);
    // Show the full guide once per browser tab, then respect the persisted
    // collapsed state for subsequent page loads.
    if (sessionStorage.getItem(sessionStartedKey) !== "1") {
      sessionStorage.setItem(sessionStartedKey, "1");
      setCollapsed(false, { useDefaultPosition: true });
      return;
    }

    const collapsed = localStorage.getItem(collapsedKey) === "1";
    guide.hidden = collapsed;
    bubble.hidden = !collapsed;
    applyStoredPosition(collapsed ? bubble : guide);
  });
})();
