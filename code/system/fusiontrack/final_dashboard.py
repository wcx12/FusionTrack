from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from fusiontrack.final_results import FinalResultsDashboard
from fusiontrack.visualization import (
    _copy_background_asset,
    _fallback_scene_size,
    _safe_name,
    _trajectory_frame_points,
)


def build_final_dashboard(
    dashboard: FinalResultsDashboard,
    output_dir: str | Path,
    fused_jsonl: str | Path | None = None,
    data_root: str | Path | None = None,
    top_sequences: int = 5,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    dashboard_data = dashboard.to_public_dict()
    playback_payloads = {}
    if fused_jsonl is not None:
        playback_payloads = _build_playback_payloads(
            dashboard=dashboard,
            fused_jsonl=Path(fused_jsonl),
            data_root=Path(data_root) if data_root is not None else Path("data") / "VT-Tiny-MOT",
            assets_dir=assets_dir,
            top_sequences=top_sequences,
        )
    (assets_dir / "final_dashboard_data.json").write_text(
        json.dumps(dashboard_data, ensure_ascii=True, separators=(",", ":")),
        encoding="utf-8",
    )
    (assets_dir / "final_playback_data.json").write_text(
        json.dumps(playback_payloads, ensure_ascii=True, separators=(",", ":")),
        encoding="utf-8",
    )
    html_text = _render_html(dashboard_data, playback_payloads)
    report_html = output_dir / "index.html"
    report_html.write_text(html_text, encoding="utf-8")
    return {
        "report_html": str(report_html),
        "assets_dir": str(assets_dir),
        "num_tasks": len(dashboard.tasks),
        "num_methods": sum(len(task.methods) for task in dashboard.tasks.values()),
        "playback_sequences": list(playback_payloads),
    }


def _build_playback_payloads(
    dashboard: FinalResultsDashboard,
    fused_jsonl: Path,
    data_root: Path,
    assets_dir: Path,
    top_sequences: int,
) -> dict[str, Any]:
    individual = dashboard.tasks.get("individual")
    if individual is None:
        return {}
    default_method = _default_method(individual.leaderboard)
    cases = individual.case_rows.get(default_method, {})
    selected_samples = []
    for case_type in ("true_positive", "false_positive", "false_negative"):
        selected_samples.extend(row["sample_id"] for row in cases.get(case_type, [])[:4])
    selected_sequences: list[str] = []
    for sample_id in selected_samples:
        sequence = sample_id.split(":", 1)[0]
        if sequence and sequence not in selected_sequences:
            selected_sequences.append(sequence)
        if len(selected_sequences) >= top_sequences:
            break
    if not selected_sequences and individual.labels:
        selected_sequences = [str(individual.labels[0].get("sequence", ""))]
    selected_set = set(selected_sequences)
    by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with fused_jsonl.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            trajectory = json.loads(stripped)
            sequence = str(trajectory.get("sequence", ""))
            if sequence in selected_set:
                by_sequence[sequence].append(trajectory)
    labels_by_sample = individual.labels_by_sample
    scores_by_method = {
        method_name: method.scores_by_sample
        for method_name, method in individual.methods.items()
    }
    payloads = {}
    for sequence in selected_sequences:
        trajectories = by_sequence.get(sequence, [])
        background_asset, background_size, background_frames = _copy_background_asset(
            sequence,
            trajectories,
            data_root,
            assets_dir,
        )
        tracks = []
        frame_ids: list[int] = []
        for trajectory in trajectories[:160]:
            frame_points = _trajectory_frame_points(trajectory)
            if not frame_points:
                continue
            frame_ids.extend(point[0] for point in frame_points)
            sample_id = str(trajectory["sample_id"])
            points = [
                {"frame": frame, "x": round(x, 3), "y": round(y, 3)}
                for frame, x, y in frame_points
            ]
            method_scores = {
                method_name: round(float(rows.get(sample_id, {}).get("score", 0.0) or 0.0), 6)
                for method_name, rows in scores_by_method.items()
            }
            label = labels_by_sample.get(sample_id, {})
            tracks.append(
                {
                    "sample_id": sample_id,
                    "sequence": sequence,
                    "track_id": str(trajectory.get("track_id", "")),
                    "category": trajectory.get("category_name", "") or "",
                    "method_scores": method_scores,
                    "label": int(label.get("label", 0) or 0),
                    "anomaly_type": str(label.get("anomaly_type", "normal")),
                    "frame_start": int(label.get("frame_start", frame_points[0][0]) or frame_points[0][0]),
                    "frame_end": int(label.get("frame_end", frame_points[-1][0]) or frame_points[-1][0]),
                    "points": points,
                }
            )
        width, height = background_size or _fallback_scene_size(trajectories)
        payloads[sequence] = {
            "sequence": sequence,
            "background": f"assets/{background_asset.name}" if background_asset else None,
            "background_frames": [
                {"frame": int(item["frame"]), "src": f"assets/{item['path'].name}"}
                for item in background_frames
            ],
            "size": {"width": width, "height": height},
            "frame_range": [min(frame_ids) if frame_ids else 0, max(frame_ids) if frame_ids else 0],
            "tracks": tracks,
        }
    return payloads


def _default_method(leaderboard: list[dict[str, Any]]) -> str:
    for row in leaderboard:
        if row.get("is_our_method"):
            return str(row["method"])
    return str(leaderboard[0]["method"]) if leaderboard else ""


def _render_html(dashboard_data: dict[str, Any], playback_payloads: dict[str, Any]) -> str:
    dashboard_json = json.dumps(dashboard_data, ensure_ascii=True).replace("</", "<\\/")
    playback_json = json.dumps(playback_payloads, ensure_ascii=True).replace("</", "<\\/")
    initial_task = "individual" if "individual" in dashboard_data["tasks"] else next(iter(dashboard_data["tasks"]), "")
    initial_method = _default_method(dashboard_data["tasks"].get(initial_task, {}).get("leaderboard", []))
    task_options = "".join(
        f'<option value="{html.escape(task)}"{ " selected" if task == initial_task else ""}>{html.escape(task.title())}</option>'
        for task in dashboard_data["tasks"]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>FusionTrack Final Results Dashboard</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Arial, sans-serif; color: #172033; background: #f6f7f9; }}
    main {{ padding: 20px 24px 36px; }}
    header {{ display: flex; justify-content: space-between; gap: 18px; align-items: end; margin-bottom: 16px; }}
    h1 {{ margin: 0; font-size: 26px; }}
    h2 {{ margin: 0 0 12px; font-size: 17px; }}
    .subtle {{ color: #64748b; font-size: 13px; }}
    .toolbar {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: end; }}
    label {{ display: grid; gap: 5px; color: #475569; font-size: 12px; }}
    select {{ min-height: 34px; min-width: 210px; border: 1px solid #cbd5e1; border-radius: 6px; padding: 6px 8px; background: white; color: #0f172a; }}
    .grid {{ display: grid; grid-template-columns: 360px 1fr; gap: 16px; align-items: start; }}
    .panel {{ background: white; border: 1px solid #e1e5eb; border-radius: 8px; padding: 14px; }}
    .side {{ position: sticky; top: 12px; max-height: calc(100vh - 24px); overflow: auto; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 10px; margin-bottom: 16px; }}
    .card {{ background: white; border: 1px solid #e1e5eb; border-radius: 8px; padding: 10px 12px; }}
    .value {{ font-size: 24px; font-weight: 700; }}
    .leaderboard, .type-table, .case-list {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px 7px; text-align: left; vertical-align: top; }}
    th {{ color: #475569; font-weight: 700; background: #f8fafc; }}
    .metric {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .badge {{ display: inline-block; border-radius: 999px; padding: 3px 8px; background: #eef2ff; color: #3730a3; font-size: 12px; }}
    .our {{ background: #ecfdf5; color: #047857; }}
    .baseline {{ background: #f8fafc; color: #475569; }}
    .method-card {{ display: grid; gap: 3px; border: 1px solid #e1e5eb; border-radius: 8px; padding: 10px; margin-bottom: 8px; cursor: pointer; background: white; text-align: left; width: 100%; }}
    .method-card.active {{ border-color: #2563eb; background: #eff6ff; }}
    .method-card strong {{ overflow-wrap: anywhere; }}
    .case-tabs {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; }}
    .case-tab {{ border: 1px solid #cbd5e1; background: white; border-radius: 999px; padding: 7px 12px; cursor: pointer; }}
    .case-tab.active {{ background: #111827; border-color: #111827; color: white; }}
    .player {{ margin-top: 16px; }}
    .canvas-shell {{ background: #111827; border-radius: 8px; padding: 10px; }}
    canvas {{ display: block; width: 100%; height: auto; background: #e2e8f0; border-radius: 6px; }}
    section {{ margin-top: 16px; }}
    @media (max-width: 960px) {{
      main {{ padding: 16px; }}
      header {{ display: grid; }}
      .grid {{ grid-template-columns: 1fr; }}
      .side {{ position: static; max-height: none; }}
      .cards {{ grid-template-columns: repeat(2, minmax(120px, 1fr)); }}
      select {{ min-width: 0; width: 100%; }}
      .toolbar {{ display: grid; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Final Results Dashboard</h1>
        <div class="subtle">Multi-method FusionTrack anomaly benchmark</div>
      </div>
      <div class="toolbar">
        <label>Task
          <select id="taskSelector">{task_options}</select>
        </label>
        <label>Method
          <select id="methodSelector"></select>
        </label>
      </div>
    </header>
    <div id="cards" class="cards"></div>
    <div class="grid">
      <aside class="panel side">
        <h2>Method leaderboard</h2>
        <div id="methodCards"></div>
      </aside>
      <div>
        <section class="panel">
          <h2>Method leaderboard</h2>
          <table class="leaderboard" id="leaderboardTable"></table>
        </section>
        <section class="panel">
          <h2>Anomaly-type analysis</h2>
          <table class="type-table" id="typeTable"></table>
        </section>
        <section class="panel">
          <h2>TP / FP / FN cases</h2>
          <div class="case-tabs">
            <button type="button" class="case-tab active" data-case="true_positive">True Positive</button>
            <button type="button" class="case-tab" data-case="false_positive">False Positive</button>
            <button type="button" class="case-tab" data-case="false_negative">False Negative</button>
          </div>
          <table class="case-list" id="caseTable"></table>
        </section>
        <section class="panel player">
          <h2>Interactive playback</h2>
          <div class="subtle" id="playbackReadout">No playback loaded</div>
          <div class="canvas-shell"><canvas id="playbackCanvas" width="960" height="612"></canvas></div>
        </section>
      </div>
    </div>
  </main>
  <script id="dashboardData" type="application/json">{dashboard_json}</script>
  <script id="playbackData" type="application/json">{playback_json}</script>
  <script>
    (() => {{
      const dashboard = JSON.parse(document.getElementById("dashboardData").textContent);
      const playbackData = JSON.parse(document.getElementById("playbackData").textContent);
      const taskSelector = document.getElementById("taskSelector");
      const methodSelector = document.getElementById("methodSelector");
      const cards = document.getElementById("cards");
      const methodCards = document.getElementById("methodCards");
      const leaderboardTable = document.getElementById("leaderboardTable");
      const typeTable = document.getElementById("typeTable");
      const caseTable = document.getElementById("caseTable");
      const caseTabs = Array.from(document.querySelectorAll(".case-tab"));
      const canvas = document.getElementById("playbackCanvas");
      const ctx = canvas.getContext("2d");
      const playbackReadout = document.getElementById("playbackReadout");
      const state = {{ task: "{html.escape(initial_task)}", method: "{html.escape(initial_method)}", caseType: "true_positive" }};

      function taskData() {{ return dashboard.tasks[state.task]; }}
      function fmt(value) {{ return Number(value || 0).toFixed(3); }}
      function esc(value) {{ return String(value ?? "").replace(/[&<>"']/g, ch => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}})[ch]); }}
      function methodsForTask(task) {{ return task.leaderboard.map(row => row.method); }}

      function setMethodOptions() {{
        const task = taskData();
        methodSelector.innerHTML = methodsForTask(task).map(method =>
          `<option value="${{esc(method)}}"${{method === state.method ? " selected" : ""}}>${{esc(method)}}</option>`
        ).join("");
      }}

      function renderCards() {{
        const task = taskData();
        const current = task.methods[state.method];
        const metrics = current.metrics;
        cards.innerHTML = [
          ["Methods", Object.keys(task.methods).length],
          ["Labels", task.num_labels],
          ["Positives", task.num_positive],
          ["Selected AUROC", fmt(metrics.auroc)]
        ].map(([label, value]) => `<div class="card"><div>${{label}}</div><div class="value">${{value}}</div></div>`).join("");
      }}

      function renderMethodCards() {{
        const task = taskData();
        methodCards.innerHTML = task.leaderboard.map(row => `
          <button type="button" class="method-card ${{row.method === state.method ? "active" : ""}}" data-method="${{esc(row.method)}}">
            <strong>${{esc(row.method)}}</strong>
            <span><span class="badge ${{row.is_our_method ? "our" : "baseline"}}">${{esc(row.owner || "method")}}</span></span>
            <span class="subtle">AUROC ${{fmt(row.auroc)}} | AUPRC ${{fmt(row.auprc)}} | F1 ${{fmt(row.f1)}}</span>
          </button>
        `).join("");
        methodCards.querySelectorAll(".method-card").forEach(button => button.addEventListener("click", () => {{
          state.method = button.dataset.method;
          renderMethodView();
        }}));
      }}

      function renderLeaderboard() {{
        const rows = taskData().leaderboard;
        leaderboardTable.innerHTML = `
          <thead><tr><th>Method</th><th>Role</th><th class="metric">AUROC</th><th class="metric">AUPRC</th><th class="metric">F1</th><th class="metric">P@100</th><th class="metric">R@100</th></tr></thead>
          <tbody>${{rows.map(row => `
            <tr>
              <td><strong>${{esc(row.method)}}</strong><br><span class="badge ${{row.is_our_method ? "our" : "baseline"}}">${{esc(row.owner || "")}}</span></td>
              <td>${{esc(row.role || row.method_family || "")}}</td>
              <td class="metric">${{fmt(row.auroc)}}</td>
              <td class="metric">${{fmt(row.auprc)}}</td>
              <td class="metric">${{fmt(row.f1)}}</td>
              <td class="metric">${{fmt(row.precision_at_k)}}</td>
              <td class="metric">${{fmt(row.recall_at_k)}}</td>
            </tr>`).join("")}}</tbody>
        `;
      }}

      function renderTypeTable() {{
        const rows = taskData().anomaly_type_rows.filter(row => row.method === state.method);
        typeTable.innerHTML = `
          <thead><tr><th>Anomaly type</th><th class="metric">Hits@K</th><th class="metric">Total</th><th class="metric">Recall@K</th><th class="metric">Mean positive score</th></tr></thead>
          <tbody>${{rows.map(row => `
            <tr><td>${{esc(row.anomaly_type)}}</td><td class="metric">${{row.hits_at_k}}</td><td class="metric">${{row.total_positive}}</td><td class="metric">${{fmt(row.recall_at_k)}}</td><td class="metric">${{fmt(row.mean_positive_score)}}</td></tr>
          `).join("")}}</tbody>
        `;
      }}

      function renderCases() {{
        const rows = ((taskData().case_rows[state.method] || {{}})[state.caseType] || []);
        caseTable.innerHTML = `
          <thead><tr><th>Sample</th><th>Type</th><th class="metric">Score</th><th class="metric">Rank</th><th>Frames</th></tr></thead>
          <tbody>${{rows.map(row => `
            <tr><td><strong>${{esc(row.sequence)}} / ${{esc(row.track_id)}}</strong><br><span class="subtle">${{esc(row.sample_id)}}</span></td><td>${{esc(row.anomaly_type)}}</td><td class="metric">${{fmt(row.score)}}</td><td class="metric">${{row.rank}}</td><td>${{row.frame_start}}-${{row.frame_end}}</td></tr>
          `).join("")}}</tbody>
        `;
      }}

      function drawPlayback() {{
        const sequences = Object.keys(playbackData);
        if (!sequences.length || state.task !== "individual") {{
          playbackReadout.textContent = "Playback is available for individual trajectories.";
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          return;
        }}
        const data = playbackData[sequences[0]];
        canvas.width = data.size.width;
        canvas.height = data.size.height;
        ctx.fillStyle = "#e2e8f0";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        const image = new Image();
        const drawTracks = () => {{
          const scores = data.tracks.map(track => Number((track.method_scores || {{}})[state.method] || 0));
          const maxScore = Math.max(...scores, 1e-6);
          const ranked = [...data.tracks].sort((a, b) => Number((b.method_scores || {{}})[state.method] || 0) - Number((a.method_scores || {{}})[state.method] || 0)).slice(0, 80);
          for (const track of ranked) {{
            const score = Number((track.method_scores || {{}})[state.method] || 0);
            const ratio = Math.max(0, Math.min(1, score / maxScore));
            ctx.strokeStyle = track.label === 1 ? "#ef4444" : `rgba(37, 99, 235, ${{0.3 + 0.55 * ratio}})`;
            ctx.lineWidth = track.label === 1 ? 3.2 : 1 + 2.5 * ratio;
            ctx.beginPath();
            track.points.forEach((point, index) => index ? ctx.lineTo(point.x, point.y) : ctx.moveTo(point.x, point.y));
            ctx.stroke();
            const last = track.points[track.points.length - 1];
            if (last) {{
              ctx.fillStyle = track.label === 1 ? "#ef4444" : "#f59e0b";
              ctx.beginPath();
              ctx.arc(last.x, last.y, track.label === 1 ? 5 : 3, 0, Math.PI * 2);
              ctx.fill();
            }}
          }}
          playbackReadout.textContent = `${{data.sequence}} / ${{state.method}} / ${{ranked.length}} visible tracks`;
        }};
        if (data.background) {{
          image.onload = () => {{ ctx.drawImage(image, 0, 0, canvas.width, canvas.height); drawTracks(); }};
          image.onerror = drawTracks;
          image.src = data.background;
        }} else {{
          drawTracks();
        }}
      }}

      function renderMethodView() {{
        setMethodOptions();
        renderCards();
        renderMethodCards();
        renderLeaderboard();
        renderTypeTable();
        renderCases();
        drawPlayback();
      }}

      taskSelector.addEventListener("change", () => {{
        state.task = taskSelector.value;
        state.method = methodsForTask(taskData())[0] || "";
        renderMethodView();
      }});
      methodSelector.addEventListener("change", () => {{
        state.method = methodSelector.value;
        renderMethodView();
      }});
      caseTabs.forEach(button => button.addEventListener("click", () => {{
        state.caseType = button.dataset.case;
        caseTabs.forEach(tab => tab.classList.toggle("active", tab === button));
        renderCases();
      }}));
      renderMethodView();
    }})();
  </script>
</body>
</html>
"""
