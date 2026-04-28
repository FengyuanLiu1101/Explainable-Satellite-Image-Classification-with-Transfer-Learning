/* ===========================================================
   EuroSAT Web Demo — Frontend logic
   -----------------------------------------------------------
   Talks to the Flask backend defined in app.py:
       POST /predict   -> { class, confidence, all_probs, gradcam_img }
       GET  /results   -> cumulative metrics.json
       GET  /health    -> server / model health check
   =========================================================== */

const CLASS_EMOJI = {
    AnnualCrop: "🌾",
    Forest: "🌳",
    HerbaceousVegetation: "🌿",
    Highway: "🛣️",
    Industrial: "🏭",
    Pasture: "🐄",
    PermanentCrop: "🌳",
    Residential: "🏘️",
    River: "🌊",
    SeaLake: "🌊",
};

const $ = (id) => document.getElementById(id);

const dropZone = $("dropZone");
const fileInput = $("fileInput");
const previewWrap = $("previewWrap");
const previewImg = $("previewImg");
const previewName = $("previewName");
const clearBtn = $("clearBtn");
const classifyBtn = $("classifyBtn");
const errorBox = $("errorBox");
const resultsCard = $("resultsCard");
const serverStatus = $("serverStatus");

let currentFile = null;
let originalDataUrl = null;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function showError(msg) {
    errorBox.textContent = msg;
    errorBox.classList.remove("hidden");
}
function clearError() {
    errorBox.textContent = "";
    errorBox.classList.add("hidden");
}
function setStatus(text, kind) {
    serverStatus.textContent = text;
    serverStatus.classList.remove("ok", "bad");
    if (kind) serverStatus.classList.add(kind);
}

// ---------------------------------------------------------------------------
// Health check on load
// ---------------------------------------------------------------------------
async function checkHealth() {
    try {
        const res = await fetch("/health");
        const data = await res.json();
        if (data.status === "ok" && data.model_loaded) {
            setStatus(`server: online · model: ready · ${data.device}`, "ok");
        } else if (data.status === "ok") {
            setStatus("server: online · model: NOT loaded (train first)", "bad");
        } else {
            setStatus("server: degraded", "bad");
        }
    } catch (e) {
        setStatus("server: offline", "bad");
    }
}

// ---------------------------------------------------------------------------
// File handling
// ---------------------------------------------------------------------------
function handleFile(file) {
    clearError();
    if (!file) return;
    if (!file.type.startsWith("image/")) {
        showError("Please upload an image file (PNG / JPG / TIFF).");
        return;
    }
    if (file.size > 10 * 1024 * 1024) {
        showError("File exceeds 10 MB limit.");
        return;
    }

    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        originalDataUrl = e.target.result;
        previewImg.src = originalDataUrl;
        previewName.textContent = `${file.name}  ·  ${(file.size / 1024).toFixed(1)} KB`;
        previewWrap.classList.remove("hidden");
        classifyBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    currentFile = null;
    originalDataUrl = null;
    fileInput.value = "";
    previewImg.src = "";
    previewName.textContent = "–";
    previewWrap.classList.add("hidden");
    classifyBtn.disabled = true;
    resultsCard.classList.add("hidden");
    clearError();
}

// Drag-and-drop
dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") fileInput.click();
});
dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    handleFile(file);
});

fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));
clearBtn.addEventListener("click", resetUpload);

// ---------------------------------------------------------------------------
// Classify
// ---------------------------------------------------------------------------
classifyBtn.addEventListener("click", async () => {
    if (!currentFile) return;
    clearError();
    classifyBtn.disabled = true;
    classifyBtn.querySelector(".btn-label").textContent = "Classifying…";

    const fd = new FormData();
    fd.append("image", currentFile);

    const t0 = performance.now();
    try {
        const res = await fetch("/predict", {
            method: "POST",
            body: fd,
        });
        const data = await res.json();
        if (!res.ok) {
            showError(data.error || `Server error (${res.status})`);
            return;
        }
        const elapsed = (performance.now() - t0).toFixed(0);
        renderResults(data, elapsed);
    } catch (err) {
        showError(`Request failed: ${err.message}`);
    } finally {
        classifyBtn.disabled = false;
        classifyBtn.querySelector(".btn-label").textContent = "Classify";
    }
});

// ---------------------------------------------------------------------------
// Render results
// ---------------------------------------------------------------------------
function renderResults(data, elapsedMs) {
    resultsCard.classList.remove("hidden");

    const cls = data.class;
    const conf = data.confidence || 0;

    $("resultEmoji").textContent =
        data.class_emoji || CLASS_EMOJI[cls] || "🛰️";
    $("resultClass").textContent = cls;
    $("resultLatency").textContent = `inference · ${elapsedMs} ms`;
    $("confidenceValue").textContent = `${(conf * 100).toFixed(1)}%`;

    // Animate confidence bar from 0 -> target
    const fill = $("confidenceFill");
    fill.style.width = "0%";
    requestAnimationFrame(() => {
        fill.style.width = `${(conf * 100).toFixed(2)}%`;
    });

    renderProbs(data.all_probs || {}, cls);

    if (originalDataUrl) {
        $("originalImg").src = originalDataUrl;
    }
    if (data.gradcam_img) {
        $("gradcamImg").src = `data:image/png;base64,${data.gradcam_img}`;
    }

    resultsCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderProbs(probs, topClass) {
    const list = $("probList");
    list.innerHTML = "";
    const entries = Object.entries(probs).sort((a, b) => b[1] - a[1]);

    for (const [name, p] of entries) {
        const row = document.createElement("div");
        row.className = "prob-row" + (name === topClass ? " top" : "");

        const nameEl = document.createElement("div");
        nameEl.className = "prob-name";
        nameEl.textContent = `${CLASS_EMOJI[name] || ""} ${name}`;

        const barWrap = document.createElement("div");
        barWrap.className = "prob-bar";
        const fillEl = document.createElement("div");
        fillEl.className = "prob-fill";
        fillEl.style.width = "0%";
        barWrap.appendChild(fillEl);

        const valEl = document.createElement("div");
        valEl.className = "prob-value";
        valEl.textContent = `${(p * 100).toFixed(1)}%`;

        row.appendChild(nameEl);
        row.appendChild(barWrap);
        row.appendChild(valEl);
        list.appendChild(row);

        // animate
        requestAnimationFrame(() => {
            fillEl.style.width = `${(p * 100).toFixed(2)}%`;
        });
    }
}

// ---------------------------------------------------------------------------
// Model comparison table
// ---------------------------------------------------------------------------
async function loadComparison() {
    const root = $("comparisonContent");
    try {
        const res = await fetch("/results");
        const data = await res.json();
        if (!data.available) {
            root.innerHTML = `<div class="loading">${
                data.message || "No metrics yet. Run training first."
            }</div>`;
            return;
        }
        const rows = data.summary || [];
        if (!rows.length) {
            root.innerHTML = `<div class="loading">metrics.json is empty.</div>`;
            return;
        }
        let bestIdx = 0;
        rows.forEach((r, i) => {
            if (r.accuracy > rows[bestIdx].accuracy) bestIdx = i;
        });

        let html = `
            <table class="compare-table">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Accuracy</th>
                        <th>Macro F1</th>
                    </tr>
                </thead>
                <tbody>
        `;
        rows.forEach((r, i) => {
            html += `
                <tr class="${i === bestIdx ? "best" : ""}">
                    <td>${r.model}</td>
                    <td>${(r.accuracy * 100).toFixed(2)}%</td>
                    <td>${(r.macro_f1 * 100).toFixed(2)}%</td>
                </tr>
            `;
        });
        html += "</tbody></table>";
        root.innerHTML = html;
    } catch (err) {
        root.innerHTML = `<div class="loading">failed to load: ${err.message}</div>`;
    }
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------
checkHealth();
loadComparison();
