let sensorChart;
const maxPoints = 60;

function makeChart() {
    const ctx = document.getElementById("sensorChart").getContext("2d");

    sensorChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: [],
            datasets: [
                {
                    label: "pH",
                    data: [],
                    borderWidth: 2,
                    tension: 0.25,
                    yAxisID: "y"
                },
                {
                    label: "Turbidity",
                    data: [],
                    borderWidth: 2,
                    tension: 0.25,
                    yAxisID: "y1"
                },
                {
                    label: "Temperature °C",
                    data: [],
                    borderWidth: 2,
                    tension: 0.25,
                    yAxisID: "y2"
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: "index", intersect: false },
            scales: {
                y: {
                    position: "left",
                    title: { display: true, text: "pH" }
                },
                y1: {
                    position: "right",
                    title: { display: true, text: "Turbidity" },
                    grid: { drawOnChartArea: false }
                },
                y2: {
                    display: false,
                    min: 0,
                    max: 50
                }
            }
        }
    });
}

function setText(id, value) {
    document.getElementById(id).textContent = value;
}

function predictionClass(prediction) {
    const p = String(prediction || "").toLowerCase();
    return p.includes("normal") && !p.includes("abnormal") ? "normal" : "abnormal";
}

function updateDashboard(payload) {
    const latest = payload.latest;
    const rows = payload.history || [];

    if (latest.ph === null) {
        setText("connectionText", "Waiting for ESP32");
        document.getElementById("statusDot").className = "dot offline";
        return;
    }

    setText("connectionText", "ESP32 connected");
    document.getElementById("statusDot").className = "dot online";

    setText("ph", latest.ph);
    setText("turbidity", latest.turbidity);
    setText("temperature", latest.temperature + " °C");

    setText("ph2", latest.ph);
    setText("turb2", latest.turbidity);
    setText("temp2", latest.temperature + " °C");

    const confidence = latest.confidence == null ? "--" : latest.confidence + "%";
    setText("confidence", "Confidence: " + confidence);
    setText("confidence2", confidence);

    const result = document.getElementById("prediction");
    result.textContent = latest.prediction || "WAITING";
    result.className = "result " + predictionClass(latest.prediction);
    setText("prediction2", latest.prediction || "--");

    setText("lastUpdated", "Last update: " + (latest.timestamp || "--"));
    setText("recordCount", rows.length + " records");

    sensorChart.data.labels = rows.map(r => r.timestamp.split(" ")[1] || r.timestamp);
    sensorChart.data.datasets[0].data = rows.map(r => r.ph);
    sensorChart.data.datasets[1].data = rows.map(r => r.turbidity);
    sensorChart.data.datasets[2].data = rows.map(r => r.temperature);
    sensorChart.update("none");

    const body = document.getElementById("historyBody");
    if (!rows.length) {
        body.innerHTML = '<tr><td colspan="6" class="empty">Waiting for sensor data...</td></tr>';
        return;
    }

    body.innerHTML = rows.slice().reverse().map(r => {
        const cls = predictionClass(r.prediction);
        const conf = r.confidence == null ? "--" : r.confidence + "%";
        return `
            <tr>
                <td>${r.timestamp}</td>
                <td>${r.ph}</td>
                <td>${r.turbidity}</td>
                <td>${r.temperature} °C</td>
                <td><span class="badge ${cls}">${r.prediction}</span></td>
                <td>${conf}</td>
            </tr>
        `;
    }).join("");
}

async function refresh() {
    try {
        const response = await fetch("/api/data", { cache: "no-store" });
        if (!response.ok) throw new Error("API error");
        const payload = await response.json();
        updateDashboard(payload);
    } catch (error) {
        setText("connectionText", "Flask server unavailable");
        document.getElementById("statusDot").className = "dot offline";
    }
}

makeChart();
refresh();
setInterval(refresh, 1000);
