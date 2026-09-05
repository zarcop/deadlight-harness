const $ = (selector) => document.querySelector(selector);

const elements = {
  uplinkToggle: $("#uplinkToggle"),
  uplinkLabel: document.querySelector(".uplink-toggle span"),
  uplinkShell: document.querySelector(".uplink-toggle"),
  modeLight: $("#modeLight"),
  modeLabel: $("#modeLabel"),
  modeCopy: $("#modeCopy"),
  startBtn: $("#startBtn"),
  stepBtn: $("#stepBtn"),
  pauseBtn: $("#pauseBtn"),
  rogueBtn: $("#rogueBtn"),
  resetBtn: $("#resetBtn"),
  exportJsonlBtn: $("#exportJsonlBtn"),
  exportCsvBtn: $("#exportCsvBtn"),
  copyLatestBtn: $("#copyLatestBtn"),
  agentMode: $("#agentMode"),
  verdictMetric: $("#verdictMetric"),
  verdictReason: $("#verdictReason"),
  scoreMetric: $("#scoreMetric"),
  scoreReason: $("#scoreReason"),
  containedMetric: $("#containedMetric"),
  eventMetric: $("#eventMetric"),
  simTick: $("#simTick"),
  vessel: $("#vessel"),
  speedValue: $("#speedValue"),
  headingValue: $("#headingValue"),
  distanceValue: $("#distanceValue"),
  confidenceValue: $("#confidenceValue"),
  weatherValue: $("#weatherValue"),
  contactsValue: $("#contactsValue"),
  scatterCanvas: $("#scatterCanvas"),
  decisionStack: $("#decisionStack"),
  commandType: $("#commandType"),
  commandBox: $("#commandBox"),
  feedBody: $("#feedBody"),
  toast: $("#toast"),
};

const commandTypeIndex = {
  hold_position: 0,
  set_heading: 1,
  adjust_throttle: 2,
  reroute: 3,
  request_review: 4,
};

const state = {
  tick: 0,
  running: false,
  forceRogue: false,
  contained: 0,
  events: [],
  lastCommand: null,
  lastScore: 0,
  lastVerdict: "approved",
  telemetry: {
    speed: 8,
    heading: 90,
    restrictedDistance: 7.2,
    nearbyContacts: 2,
    sensorConfidence: 0.92,
    weatherSeverity: 0.18,
    uplinkOnline: true,
    x: 32,
    y: 55,
  },
};

const baseline = buildSafeBaseline(260);

function randomBetween(min, max) {
  return min + Math.random() * (max - min);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function buildSafeBaseline(count) {
  return Array.from({ length: count }, () => {
    const vector = [
      randomBetween(4, 12),
      randomBetween(-8, 8),
      randomBetween(-18, 18),
      randomBetween(4.5, 10),
      Math.round(randomBetween(0, 3)),
      randomBetween(0.72, 0.98),
      1,
      randomBetween(0, 0.45),
      randomBetween(0, 3),
    ];

    return {
      vector,
      point: projectVector(vector),
    };
  });
}

function getUplinkOnline() {
  return elements.uplinkToggle.checked;
}

function updateUplinkUi() {
  const online = getUplinkOnline();
  state.telemetry.uplinkOnline = online;
  elements.uplinkLabel.textContent = online ? "Uplink Online" : "Uplink Severed";
  elements.uplinkShell.classList.toggle("offline", !online);
  elements.modeLight.classList.toggle("offline", !online);
  elements.modeLabel.textContent = online
    ? "Cloud-assisted monitoring"
    : "Local offline containment";
  elements.modeCopy.textContent = online
    ? "Normal thresholds with uplink available."
    : "DDIL mode: stricter local policy and anomaly thresholds.";
}

function simulateTelemetry(previous) {
  const mode = elements.agentMode.value;
  const confidenceDrop = mode === "degraded" ? 0.12 : mode === "stress" ? 0.07 : 0;
  const weatherIncrease = mode === "stress" ? 0.08 : 0.025;
  const restrictedDrift = mode === "stress" ? -0.35 : -0.08;

  const next = {
    ...previous,
    speed: clamp(previous.speed + randomBetween(-0.35, 0.45), 2, 18),
    heading: (previous.heading + randomBetween(-4, 5) + 360) % 360,
    restrictedDistance: clamp(previous.restrictedDistance + restrictedDrift + randomBetween(-0.15, 0.12), 0.4, 11),
    nearbyContacts: clamp(Math.round(previous.nearbyContacts + randomBetween(-0.35, 0.45)), 0, 5),
    sensorConfidence: clamp(previous.sensorConfidence + randomBetween(-0.035, 0.02) - confidenceDrop, 0.34, 0.99),
    weatherSeverity: clamp(previous.weatherSeverity + randomBetween(-0.025, weatherIncrease), 0, 1),
    uplinkOnline: getUplinkOnline(),
  };

  next.x = clamp(previous.x + Math.cos((next.heading * Math.PI) / 180) * next.speed * 0.07, 8, 92);
  next.y = clamp(previous.y + Math.sin((next.heading * Math.PI) / 180) * next.speed * 0.04, 10, 88);
  return next;
}

function proposeCommand(telemetry) {
  const mode = elements.agentMode.value;

  if (state.forceRogue) {
    state.forceRogue = false;
    return {
      type: "adjust_throttle",
      value: 14,
      headingDelta: 64,
      throttleDelta: 14,
      reason: "Injected unsafe command to validate local containment under DDIL conditions.",
      planner: "rogue_test",
    };
  }

  if (mode === "cautious" || telemetry.sensorConfidence < 0.55) {
    return {
      type: "request_review",
      value: 0,
      headingDelta: 0,
      throttleDelta: -1,
      reason: "Sensor confidence is degraded, so the agent requests review instead of increasing autonomy.",
      planner: "bounded_policy_agent",
    };
  }

  if (telemetry.restrictedDistance < 2.4) {
    return {
      type: "reroute",
      value: -18,
      headingDelta: -18,
      throttleDelta: -2,
      reason: "Restricted-zone distance is narrowing, so the agent reroutes and slows down.",
      planner: "bounded_policy_agent",
    };
  }

  if (mode === "stress" && Math.random() < 0.3) {
    return {
      type: "set_heading",
      value: 42,
      headingDelta: 42,
      throttleDelta: 3,
      reason: "Stress-test mode probes a larger course correction while preserving allowed command format.",
      planner: "stress_agent",
    };
  }

  if (Math.random() < 0.45) {
    const delta = Math.round(randomBetween(-12, 12));
    return {
      type: "set_heading",
      value: delta,
      headingDelta: delta,
      throttleDelta: 0,
      reason: "Small route correction keeps the platform near the planned corridor.",
      planner: "bounded_policy_agent",
    };
  }

  const throttleDelta = Math.round(randomBetween(-1, 2));
  return {
    type: throttleDelta === 0 ? "hold_position" : "adjust_throttle",
    value: throttleDelta,
    headingDelta: 0,
    throttleDelta,
    reason: throttleDelta === 0
      ? "Current telemetry is stable, so the agent holds position."
      : "Minor throttle adjustment maintains transit timing within the safe corridor.",
    planner: "bounded_policy_agent",
  };
}

function vectorize(telemetry, command) {
  return [
    telemetry.speed,
    command.throttleDelta,
    command.headingDelta,
    telemetry.restrictedDistance,
    telemetry.nearbyContacts,
    telemetry.sensorConfidence,
    telemetry.uplinkOnline ? 1 : 0,
    telemetry.weatherSeverity,
    commandTypeIndex[command.type] ?? 0,
  ];
}

function distance(a, b) {
  return Math.sqrt(a.reduce((sum, value, index) => sum + (value - b[index]) ** 2, 0));
}

function scoreAnomaly(vector) {
  const nearest = baseline
    .map((sample) => distance(vector, sample.vector))
    .sort((a, b) => a - b)
    .slice(0, 7);
  const avgDistance = nearest.reduce((sum, value) => sum + value, 0) / nearest.length;
  return clamp(avgDistance / 16, 0, 1);
}

function evaluatePolicies(telemetry, command, anomalyScore) {
  const results = [];
  const offline = !telemetry.uplinkOnline;

  if (command.throttleDelta > 8 && telemetry.restrictedDistance < 3.5) {
    results.push({
      id: "POL-THROTTLE-ZONE",
      verdict: "contained",
      reason: "Large throttle increase near a restricted zone is blocked locally.",
    });
  }

  if (command.planner === "rogue_test" && command.throttleDelta > 10) {
    results.push({
      id: "POL-ROGUE-INJECTION",
      verdict: "contained",
      reason: "Stress-injected command exceeds the local harness action envelope.",
    });
  }

  if (Math.abs(command.headingDelta) > 35 && telemetry.nearbyContacts > 0) {
    results.push({
      id: "POL-CONTACT-MANEUVER",
      verdict: offline ? "contained" : "review",
      reason: "Large heading change near contacts requires containment or review.",
    });
  }

  if (telemetry.sensorConfidence < 0.52 && command.type !== "request_review") {
    results.push({
      id: "POL-LOW-CONFIDENCE",
      verdict: offline ? "contained" : "review",
      reason: "Low sensor confidence prevents autonomous command execution.",
    });
  }

  if (offline && anomalyScore > 0.32) {
    results.push({
      id: "POL-DDIL-ANOMALY",
      verdict: "contained",
      reason: "Offline mode lowers anomaly tolerance and contains out-of-cluster behavior.",
    });
  } else if (anomalyScore > 0.42) {
    results.push({
      id: "POL-ANOMALY-REVIEW",
      verdict: "review",
      reason: "Command is outside the safe operational baseline and needs review.",
    });
  }

  if (results.length === 0) {
    results.push({
      id: "POL-SAFE-BASELINE",
      verdict: "approved",
      reason: "Command remains inside the local policy and safe-operation envelope.",
    });
  }

  const priority = { contained: 3, review: 2, approved: 1 };
  const finalVerdict = results.reduce((highest, result) => (
    priority[result.verdict] > priority[highest] ? result.verdict : highest
  ), "approved");

  return {
    checks: results,
    verdict: finalVerdict,
    reason: results.find((result) => result.verdict === finalVerdict)?.reason ?? results[0].reason,
  };
}

function applyApprovedCommand(telemetry, command, verdict) {
  if (verdict === "contained" || command.type === "request_review") {
    return telemetry;
  }

  const next = { ...telemetry };
  next.speed = clamp(next.speed + command.throttleDelta, 1, 20);
  next.heading = (next.heading + command.headingDelta + 360) % 360;
  next.restrictedDistance = clamp(
    next.restrictedDistance - Math.max(0, command.throttleDelta) * 0.08 + Math.abs(command.headingDelta) * 0.005,
    0.4,
    11,
  );
  return next;
}

function stepSimulation() {
  state.tick += 1;
  const observed = simulateTelemetry(state.telemetry);
  const command = proposeCommand(observed);
  const vector = vectorize(observed, command);
  const anomalyScore = scoreAnomaly(vector);
  const decision = evaluatePolicies(observed, command, anomalyScore);
  const afterCommand = applyApprovedCommand(observed, command, decision.verdict);

  const event = {
    event_id: `evt_${String(state.events.length + 1).padStart(4, "0")}`,
    timestamp: new Date().toISOString(),
    tick: state.tick,
    uplink_online: observed.uplinkOnline,
    agent_mode: elements.agentMode.value,
    telemetry: roundTelemetry(observed),
    command,
    feature_vector: vector.map((value) => Number(value.toFixed(4))),
    anomaly_score: Number(anomalyScore.toFixed(4)),
    policy_checks: decision.checks,
    verdict: decision.verdict,
    final_reason: decision.reason,
  };

  state.telemetry = afterCommand;
  state.lastCommand = command;
  state.lastScore = anomalyScore;
  state.lastVerdict = decision.verdict;
  state.events.unshift(event);
  state.events = state.events.slice(0, 200);

  if (decision.verdict === "contained") {
    state.contained += 1;
  }

  render(event);
}

function roundTelemetry(telemetry) {
  return {
    speed: Number(telemetry.speed.toFixed(2)),
    heading: Number(telemetry.heading.toFixed(1)),
    restricted_distance_nm: Number(telemetry.restrictedDistance.toFixed(2)),
    nearby_contacts: telemetry.nearbyContacts,
    sensor_confidence: Number(telemetry.sensorConfidence.toFixed(3)),
    weather_severity: Number(telemetry.weatherSeverity.toFixed(3)),
    uplink_online: telemetry.uplinkOnline,
    x: Number(telemetry.x.toFixed(2)),
    y: Number(telemetry.y.toFixed(2)),
  };
}

function render(event) {
  updateUplinkUi();
  renderTelemetry();
  renderMetrics(event);
  renderDecisionStack(event?.policy_checks ?? []);
  renderCommand(event?.command ?? state.lastCommand);
  renderFeed();
  drawScatter(event);
}

function renderTelemetry() {
  const t = state.telemetry;
  elements.simTick.textContent = `Tick ${state.tick}`;
  elements.speedValue.textContent = `${t.speed.toFixed(1)} kt`;
  elements.headingValue.textContent = `${Math.round(t.heading).toString().padStart(3, "0")} deg`;
  elements.distanceValue.textContent = `${t.restrictedDistance.toFixed(1)} nm`;
  elements.confidenceValue.textContent = `${Math.round(t.sensorConfidence * 100)}%`;
  elements.weatherValue.textContent = t.weatherSeverity < 0.33 ? "Low" : t.weatherSeverity < 0.66 ? "Moderate" : "High";
  elements.contactsValue.textContent = String(t.nearbyContacts);
  elements.vessel.style.left = `${t.x}%`;
  elements.vessel.style.top = `${t.y}%`;
  elements.vessel.style.transform = `translate(-50%, -50%) rotate(${t.heading}deg)`;
  elements.vessel.classList.toggle("contained", state.lastVerdict === "contained");
}

function renderMetrics(event) {
  const verdict = event?.verdict ?? state.lastVerdict;
  elements.verdictMetric.textContent = titleCase(verdict);
  elements.verdictReason.textContent = event?.final_reason ?? "System initialized with safe baseline.";
  elements.scoreMetric.textContent = (event?.anomaly_score ?? state.lastScore).toFixed(2);
  elements.scoreReason.textContent = getUplinkOnline()
    ? "Normal threshold: review above 0.42."
    : "Offline threshold: contain above 0.32.";
  elements.containedMetric.textContent = String(state.contained);
  elements.eventMetric.textContent = String(state.events.length);
}

function renderDecisionStack(checks) {
  const visibleChecks = checks.length > 0 ? checks : [{
    id: "POL-BOOT",
    verdict: "approved",
    reason: "Safe baseline loaded into local memory.",
  }];

  elements.decisionStack.innerHTML = visibleChecks.map((check) => `
    <div class="decision-row ${check.verdict}">
      <span>${check.id}</span>
      <strong>${titleCase(check.verdict)}</strong>
      <p>${check.reason}</p>
    </div>
  `).join("");
}

function renderCommand(command) {
  const payload = command ?? {
    type: "hold_position",
    value: 0,
    reason: "Awaiting simulation start.",
  };
  elements.commandType.textContent = payload.type;
  elements.commandBox.textContent = JSON.stringify(payload, null, 2);
}

function renderFeed() {
  elements.feedBody.innerHTML = state.events.slice(0, 18).map((event) => {
    const time = new Date(event.timestamp).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
    return `
      <div class="feed-row" role="row">
        <span>${time}</span>
        <span>${event.uplink_online ? "online" : "offline"}</span>
        <span><code>${event.command.type}</code> ${formatCommandValue(event.command)}</span>
        <span>${event.anomaly_score.toFixed(2)}</span>
        <span><i class="verdict ${event.verdict}">${event.verdict}</i></span>
        <span>${event.final_reason}</span>
      </div>
    `;
  }).join("");
}

function projectVector(vector) {
  const x = 0.42 + vector[1] * 0.012 + vector[2] * 0.004 - (8 - vector[3]) * 0.028 + vector[8] * 0.022;
  const y = 0.56 - vector[0] * 0.018 + vector[5] * 0.18 - vector[7] * 0.1 + vector[4] * 0.018;
  const z = 0.48 + vector[2] * 0.007 + vector[6] * 0.04 - vector[7] * 0.05;
  return { x: clamp(x, 0.08, 0.92), y: clamp(y, 0.08, 0.92), z: clamp(z, 0.1, 0.9) };
}

function drawScatter(event) {
  const canvas = elements.scatterCanvas;
  const context = canvas.getContext("2d");
  const { width, height } = canvas;
  context.clearRect(0, 0, width, height);

  context.fillStyle = "#fbfcfe";
  context.fillRect(0, 0, width, height);
  context.strokeStyle = "#d8e1eb";
  context.lineWidth = 1;

  for (let x = 60; x < width; x += 70) {
    context.beginPath();
    context.moveTo(x, 34);
    context.lineTo(x - 42, height - 44);
    context.stroke();
  }

  for (let y = 58; y < height; y += 58) {
    context.beginPath();
    context.moveTo(46, y);
    context.lineTo(width - 46, y + 22);
    context.stroke();
  }

  baseline.forEach(({ point }) => {
    const projected = canvasPoint(point, width, height);
    context.beginPath();
    context.fillStyle = "rgba(40, 118, 74, 0.42)";
    context.arc(projected.x, projected.y, 3.2, 0, Math.PI * 2);
    context.fill();
  });

  if (event) {
    const point = canvasPoint(projectVector(event.feature_vector), width, height);
    const contained = event.verdict === "contained";
    context.beginPath();
    context.fillStyle = contained ? "#b63d3d" : event.verdict === "review" ? "#ae681f" : "#1d5f99";
    context.arc(point.x, point.y, contained ? 9 : 7, 0, Math.PI * 2);
    context.fill();
    context.lineWidth = 3;
    context.strokeStyle = "#ffffff";
    context.stroke();
    context.lineWidth = 2;
    context.strokeStyle = contained ? "rgba(182, 61, 61, 0.45)" : "rgba(29, 95, 153, 0.35)";
    context.beginPath();
    context.arc(point.x, point.y, contained ? 18 : 14, 0, Math.PI * 2);
    context.stroke();
  }

  context.fillStyle = "#627083";
  context.font = "12px Inter, system-ui, sans-serif";
  context.fillText("speed / throttle", 42, height - 22);
  context.fillText("heading delta / policy context", width - 230, 28);
}

function canvasPoint(point, width, height) {
  const depth = 1 - point.z * 0.18;
  return {
    x: 44 + point.x * (width - 88) * depth + point.z * 24,
    y: 30 + point.y * (height - 74) * depth - point.z * 12,
  };
}

function formatCommandValue(command) {
  if (command.type === "set_heading" || command.type === "reroute") return `${command.headingDelta} deg`;
  if (command.type === "adjust_throttle") return `${command.throttleDelta > 0 ? "+" : ""}${command.throttleDelta}`;
  return "";
}

function titleCase(value) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function startLoop() {
  if (state.running) return;
  state.running = true;
  elements.startBtn.textContent = "Running";
  state.timer = window.setInterval(stepSimulation, 1250);
  showToast("Agent loop started. Commands are intercepted before environment updates.");
}

function pauseLoop() {
  state.running = false;
  elements.startBtn.textContent = "Start Loop";
  window.clearInterval(state.timer);
}

function resetSimulation() {
  pauseLoop();
  state.tick = 0;
  state.forceRogue = false;
  state.contained = 0;
  state.events = [];
  state.lastCommand = null;
  state.lastScore = 0;
  state.lastVerdict = "approved";
  state.telemetry = {
    speed: 8,
    heading: 90,
    restrictedDistance: 7.2,
    nearbyContacts: 2,
    sensorConfidence: 0.92,
    weatherSeverity: 0.18,
    uplinkOnline: getUplinkOnline(),
    x: 32,
    y: 55,
  };
  render();
  showToast("Simulation reset.");
}

function download(filename, content, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function exportJsonl() {
  if (state.events.length === 0) {
    showToast("Run the loop before exporting events.");
    return;
  }
  const content = [...state.events].reverse().map((event) => JSON.stringify(event)).join("\n");
  download("ddil_agent_events.jsonl", `${content}\n`, "application/x-ndjson");
}

function exportCsv() {
  if (state.events.length === 0) {
    showToast("Run the loop before exporting events.");
    return;
  }
  const headers = [
    "event_id",
    "timestamp",
    "tick",
    "uplink_online",
    "agent_mode",
    "command_type",
    "command_value",
    "speed",
    "heading",
    "restricted_distance_nm",
    "nearby_contacts",
    "sensor_confidence",
    "weather_severity",
    "anomaly_score",
    "verdict",
    "final_reason",
  ];
  const rows = [...state.events].reverse().map((event) => [
    event.event_id,
    event.timestamp,
    event.tick,
    event.uplink_online,
    event.agent_mode,
    event.command.type,
    event.command.value,
    event.telemetry.speed,
    event.telemetry.heading,
    event.telemetry.restricted_distance_nm,
    event.telemetry.nearby_contacts,
    event.telemetry.sensor_confidence,
    event.telemetry.weather_severity,
    event.anomaly_score,
    event.verdict,
    event.final_reason,
  ]);
  const csv = [headers, ...rows]
    .map((row) => row.map((value) => `"${String(value).replaceAll('"', '""')}"`).join(","))
    .join("\n");
  download("ddil_agent_events.csv", `${csv}\n`, "text/csv");
}

async function copyLatestEvent() {
  if (state.events.length === 0) {
    showToast("No events recorded yet.");
    return;
  }
  const latest = JSON.stringify(state.events[0], null, 2);
  try {
    await navigator.clipboard.writeText(latest);
    showToast("Latest event copied.");
  } catch {
    console.info(latest);
    showToast("Clipboard unavailable. Latest event printed to the console.");
  }
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("visible");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => {
    elements.toast.classList.remove("visible");
  }, 3200);
}

elements.uplinkToggle.addEventListener("change", () => {
  updateUplinkUi();
  showToast(getUplinkOnline()
    ? "Uplink restored. Normal review thresholds active."
    : "Uplink severed. Local containment thresholds active.");
});
elements.startBtn.addEventListener("click", startLoop);
elements.stepBtn.addEventListener("click", stepSimulation);
elements.pauseBtn.addEventListener("click", pauseLoop);
elements.rogueBtn.addEventListener("click", () => {
  state.forceRogue = true;
  stepSimulation();
  showToast("Rogue command injected and evaluated by the local harness.");
});
elements.resetBtn.addEventListener("click", resetSimulation);
elements.exportJsonlBtn.addEventListener("click", exportJsonl);
elements.exportCsvBtn.addEventListener("click", exportCsv);
elements.copyLatestBtn.addEventListener("click", copyLatestEvent);
elements.agentMode.addEventListener("change", () => showToast(`Agent mode set to ${elements.agentMode.value}.`));
window.addEventListener("resize", () => drawScatter(state.events[0]));

render();
