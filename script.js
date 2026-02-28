const GRID_SIZE = 8;
const ACTIONS = [
  { name: "up", delta: [0, -1] },
  { name: "right", delta: [1, 0] },
  { name: "down", delta: [0, 1] },
  { name: "left", delta: [-1, 0] },
];

const environment = {
  gridSize: GRID_SIZE,
  walls: new Set([
    "1,1",
    "1,2",
    "1,3",
    "2,3",
    "3,3",
    "5,0",
    "5,1",
    "5,2",
    "5,3",
    "5,4",
  ]),
  hellStates: new Set(["3,5", "2,6", "4,6", "6,4"]),
  teleporters: new Map([
    ["0,5", { to: [4, 1] }],
    ["6,2", { to: [2, 6] }],
  ]),
  start: [0, 0],
  goal: [7, 7],
  livingReward: -0.01,
  goalReward: 10,
  hellReward: -5,
};

const qTable = new Map();
let trainingRewards = [];
let rewardChart;

// IndexedDB persistence
const DB_NAME = "QlearningDB";
const DB_VERSION = 1;
const STORE_NAME = "qtable";
const QTABLE_KEY = "qTable";

const dbPromise = new Promise((resolve, reject) => {
  const request = indexedDB.open(DB_NAME, DB_VERSION);
  request.onupgradeneeded = (event) => {
    event.target.result.createObjectStore(STORE_NAME);
  };
  request.onsuccess = (event) => resolve(event.target.result);
  request.onerror = (event) => reject(event.target.error);
});

function txRequest(idbRequest) {
  return new Promise((resolve, reject) => {
    idbRequest.onsuccess = (e) => resolve(e.target.result);
    idbRequest.onerror = (e) => reject(e.target.error);
  });
}

function txWrite(fn) {
  return dbPromise.then((db) =>
    new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readwrite");
      tx.oncomplete = resolve;
      tx.onerror = (e) => reject(e.target.error);
      fn(tx.objectStore(STORE_NAME));
    })
  );
}

async function saveQTable() {
  try {
    await txWrite((store) => store.put(Array.from(qTable.entries()), QTABLE_KEY));
  } catch (err) {
    console.warn("Failed to save Q-table:", err);
  }
}

async function loadSavedQTable() {
  try {
    const db = await dbPromise;
    const tx = db.transaction(STORE_NAME, "readonly");
    const serialized = await txRequest(tx.objectStore(STORE_NAME).get(QTABLE_KEY));
    if (serialized && serialized.length > 0) {
      qTable.clear();
      for (const [key, values] of serialized) {
        qTable.set(key, values);
      }
      return true;
    }
  } catch (err) {
    console.warn("Failed to load Q-table:", err);
  }
  return false;
}

async function clearPersistedQTable() {
  try {
    await txWrite((store) => store.clear());
  } catch (err) {
    console.warn("Failed to clear Q-table:", err);
  }
}

function stateKey([x, y]) {
  return `${x},${y}`;
}

function getQValues(state) {
  const key = stateKey(state);
  if (!qTable.has(key)) {
    qTable.set(key, new Array(ACTIONS.length).fill(0));
  }
  return qTable.get(key);
}

function isInsideGrid([x, y]) {
  return x >= 0 && y >= 0 && x < GRID_SIZE && y < GRID_SIZE;
}

function isWall([x, y]) {
  return environment.walls.has(stateKey([x, y]));
}

function getNextState(state, actionIndex) {
  const [dx, dy] = ACTIONS[actionIndex].delta;
  const tentative = [state[0] + dx, state[1] + dy];

  if (!isInsideGrid(tentative) || isWall(tentative)) {
    return [...state];
  }

  const teleporter = environment.teleporters.get(stateKey(tentative));
  if (teleporter) {
    return [...teleporter.to];
  }

  return tentative;
}

function getReward(state) {
  if (stateKey(state) === stateKey(environment.goal)) {
    return environment.goalReward;
  }
  if (environment.hellStates.has(stateKey(state))) {
    return environment.hellReward;
  }
  return environment.livingReward;
}

function isTerminal(state) {
  return (
    stateKey(state) === stateKey(environment.goal) ||
    environment.hellStates.has(stateKey(state))
  );
}

function greedyAction(state) {
  const qValues = getQValues(state);
  let maxIndex = 0;
  let maxValue = qValues[0];
  for (let i = 1; i < qValues.length; i += 1) {
    if (qValues[i] > maxValue) {
      maxValue = qValues[i];
      maxIndex = i;
    }
  }
  return maxIndex;
}

function epsilonGreedyAction(state, epsilon) {
  if (Math.random() < epsilon) {
    return Math.floor(Math.random() * ACTIONS.length);
  }
  return greedyAction(state);
}

async function trainAgent(options) {
  const {
    episodes,
    alpha,
    gamma,
    epsilonStart,
    epsilonEnd,
    maxSteps,
    statusNode,
  } = options;

  trainingRewards = [];
  const epsilonDecay = Math.pow(epsilonEnd / epsilonStart || 0.0001, 1 / episodes);
  let epsilon = epsilonStart;

  for (let episode = 0; episode < episodes; episode += 1) {
    let state = [...environment.start];
    let totalReward = 0;

    for (let step = 0; step < maxSteps; step += 1) {
      const action = epsilonGreedyAction(state, epsilon);
      const nextState = getNextState(state, action);
      const reward = getReward(nextState);
      const done = isTerminal(nextState);

      const qValues = getQValues(state);
      const nextQValues = getQValues(nextState);

      qValues[action] =
        qValues[action] +
        alpha * (reward + gamma * Math.max(...nextQValues) - qValues[action]);

      state = nextState;
      totalReward += reward;

      if (done) {
        break;
      }
    }

    trainingRewards.push(totalReward);
    epsilon = Math.max(epsilon * epsilonDecay, epsilonEnd);

    if (episode % Math.max(1, Math.floor(episodes / 20)) === 0) {
      statusNode.textContent = `Episode ${episode + 1} / ${episodes} · ε=${epsilon
        .toFixed(3)} · Latest reward: ${totalReward.toFixed(2)}`;
      await new Promise((resolve) => requestAnimationFrame(resolve));
    }
  }

  statusNode.textContent = `Training complete! ε=${epsilon.toFixed(3)} · Last reward: ${
    trainingRewards.at(-1)?.toFixed(2) ?? "0.00"
  }`;
}

function drawGrid(agentPosition = environment.start) {
  const grid = document.getElementById("grid");
  grid.innerHTML = "";

  for (let y = 0; y < GRID_SIZE; y += 1) {
    for (let x = 0; x < GRID_SIZE; x += 1) {
      const cell = document.createElement("div");
      cell.className = "cell";
      cell.setAttribute("role", "gridcell");
      cell.setAttribute("aria-label", `Cell ${x},${y}`);

      const key = stateKey([x, y]);
      if (stateKey([x, y]) === stateKey(agentPosition)) {
        cell.classList.add("agent", "active");
        cell.innerHTML = "<span>🐭</span>";
      } else if (stateKey([x, y]) === stateKey(environment.goal)) {
        cell.classList.add("goal");
        cell.innerHTML = "<span>🏁</span>";
      } else if (environment.walls.has(key)) {
        cell.classList.add("wall");
        cell.innerHTML = "<span>🧱</span>";
      } else if (environment.hellStates.has(key)) {
        cell.classList.add("hell");
        cell.innerHTML = "<span>🔥</span>";
      } else if (environment.teleporters.has(key)) {
        cell.classList.add("teleporter");
        cell.innerHTML = "<span>🌀</span>";
      }

      grid.appendChild(cell);
    }
  }
}

function updateMetrics() {
  if (!rewardChart) {
    const ctx = document.getElementById("reward-chart").getContext("2d");
    rewardChart = new Chart(ctx, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: "Episode Reward",
            data: [],
            borderColor: "#2563eb",
            backgroundColor: "rgba(37, 99, 235, 0.2)",
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.25,
          },
        ],
      },
      options: {
        responsive: true,
        scales: {
          x: { title: { display: true, text: "Episode" } },
          y: { title: { display: true, text: "Total Reward" } },
        },
      },
    });
  }

  rewardChart.data.labels = trainingRewards.map((_, index) => index + 1);
  rewardChart.data.datasets[0].data = trainingRewards;
  rewardChart.update();

  const metricsSummary = document.getElementById("metrics-summary");
  if (trainingRewards.length === 0) {
    metricsSummary.textContent = "Train the agent to populate metrics.";
    return;
  }

  const lastHundred = trainingRewards.slice(-100);
  const average =
    trainingRewards.reduce((sum, value) => sum + value, 0) / trainingRewards.length;
  const averageRecent =
    lastHundred.reduce((sum, value) => sum + value, 0) / lastHundred.length;
  const bestReward = Math.max(...trainingRewards).toFixed(2);

  metricsSummary.innerHTML = `
    <div>Episodes: <strong>${trainingRewards.length}</strong></div>
    <div>Average Reward: <strong>${average.toFixed(2)}</strong></div>
    <div>Average (Last 100): <strong>${averageRecent.toFixed(2)}</strong></div>
    <div>Best Reward: <strong>${bestReward}</strong></div>
  `;
}

async function playGreedyRun() {
  const runStatus = document.getElementById("run-status");
  runStatus.textContent = "Running greedy policy…";
  let state = [...environment.start];
  drawGrid(state);

  const visited = new Set();

  for (let step = 0; step < 200; step += 1) {
    await new Promise((resolve) => setTimeout(resolve, 300));

    const action = greedyAction(state);
    const nextState = getNextState(state, action);
    const reward = getReward(nextState);
    drawGrid(nextState);

    if (isTerminal(nextState)) {
      runStatus.textContent =
        reward === environment.goalReward
          ? `Goal reached in ${step + 1} steps!`
          : `Terminated in hell state after ${step + 1} steps.`;
      return;
    }

    const key = stateKey(nextState);
    if (visited.has(key)) {
      runStatus.textContent =
        "Agent entered a loop. Continue training to improve the policy.";
      return;
    }
    visited.add(key);

    state = nextState;
  }

  runStatus.textContent =
    "Max steps reached without finding the goal. Consider additional training.";
}

function resetEnvironment() {
  drawGrid(environment.start);
  document.getElementById("run-status").textContent = "";
}

document.addEventListener("DOMContentLoaded", async () => {
  const statusNode = document.getElementById("status");
  const playButton = document.getElementById("play-button");
  const trainButton = document.getElementById("train-button");
  const resetButton = document.getElementById("reset-button");

  drawGrid(environment.start);

  const loaded = await loadSavedQTable();
  if (loaded) {
    statusNode.textContent = `Restored Q-table from a previous session (${qTable.size} states). Ready to run or continue training.`;
    playButton.disabled = false;
  }

  updateMetrics();

  document
    .getElementById("training-form")
    .addEventListener("submit", async (event) => {
      event.preventDefault();
      playButton.disabled = true;
      trainButton.disabled = true;
      resetButton.disabled = true;
      statusNode.textContent = "Training in progress…";
      document.getElementById("run-status").textContent = "";

      const episodes = Number(document.getElementById("episodes").value);
      const alpha = Number(document.getElementById("alpha").value);
      const gamma = Number(document.getElementById("gamma").value);
      const epsilonStart = Number(
        document.getElementById("epsilon-start").value
      );
      const epsilonEnd = Number(document.getElementById("epsilon-end").value);
      const maxSteps = Number(document.getElementById("max-steps").value);

      await trainAgent({
        episodes,
        alpha,
        gamma,
        epsilonStart,
        epsilonEnd,
        maxSteps,
        statusNode,
      });

      await saveQTable();
      updateMetrics();
      resetEnvironment();
      playButton.disabled = false;
      trainButton.disabled = false;
      resetButton.disabled = false;
    });

  playButton.addEventListener("click", () => {
    playButton.disabled = true;
    resetButton.disabled = true;
    playGreedyRun().finally(() => {
      playButton.disabled = false;
      resetButton.disabled = false;
    });
  });

  resetButton.addEventListener("click", async () => {
    resetButton.disabled = true;
    await clearPersistedQTable();
    qTable.clear();
    trainingRewards = [];
    playButton.disabled = true;
    statusNode.textContent = "Q-table cleared. Waiting to start training…";
    updateMetrics();
    resetEnvironment();
    resetButton.disabled = false;
  });
});
