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
    labels_by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for label in individual.labels:
        labels_by_sequence[str(label.get("sequence", ""))].append(label)
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
        sequence_labels = labels_by_sequence.get(sequence, [])
        frame_start = min(frame_ids) if frame_ids else 0
        frame_end = max(frame_ids) if frame_ids else 0
        payloads[sequence] = {
            "sequence": sequence,
            "background": f"assets/{background_asset.name}" if background_asset else None,
            "background_frames": [
                {"frame": int(item["frame"]), "src": f"assets/{item['path'].name}"}
                for item in background_frames
            ],
            "size": {"width": width, "height": height},
            "frame_range": [frame_start, frame_end],
            "stats": {
                "sequence_sample_count": len(sequence_labels) if sequence_labels else len(tracks),
                "sequence_anomaly_count": sum(1 for row in sequence_labels if int(row.get("label", 0) or 0) == 1),
                "frame_start": frame_start,
                "frame_end": frame_end,
                "visualized_tracks": len(tracks),
            },
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
    initial_sequence = next(iter(playback_payloads), "")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FusionTrack 最终结果看板</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Arial, sans-serif; color: #172033; background: #f5f7fb; line-height: 1.5; }}
    main {{ max-width: 1500px; margin: 0 auto; padding: 24px 24px 40px; }}
    header {{ display: flex; justify-content: space-between; gap: 18px; align-items: flex-start; margin-bottom: 16px; }}
    h1 {{ margin: 0; font-size: 28px; line-height: 1.15; }}
    h2 {{ margin: 0; font-size: 18px; line-height: 1.25; }}
    .subtle {{ color: #64748b; font-size: 13px; }}
    .toolbar {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: end; justify-content: flex-end; }}
    label {{ display: grid; gap: 5px; color: #475569; font-size: 12px; font-weight: 700; }}
    select, button, input {{ min-height: 44px; border: 1px solid #cbd5e1; border-radius: 7px; padding: 8px 10px; background: white; color: #0f172a; }}
    select {{ min-width: 170px; }}
    button {{ cursor: pointer; font-weight: 700; transition: background-color 160ms ease, border-color 160ms ease, color 160ms ease, transform 160ms ease; touch-action: manipulation; }}
    button:hover {{ border-color: #94a3b8; background: #f8fafc; }}
    button:active {{ transform: translateY(1px); }}
    select:focus-visible, button:focus-visible, input:focus-visible {{ outline: 3px solid rgba(14, 116, 144, 0.28); outline-offset: 2px; }}
    input[type="range"] {{ min-height: 32px; padding: 0; accent-color: #0f766e; cursor: pointer; }}
    .panel {{ background: white; border: 1px solid #e1e7ef; border-radius: 8px; padding: 16px; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04); }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 10px; margin-bottom: 16px; }}
    .card {{ background: white; border: 1px solid #e1e7ef; border-radius: 8px; padding: 11px 13px; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.035); }}
    .card > div:first-child {{ color: #64748b; font-size: 12px; font-weight: 700; }}
    .value {{ margin-top: 2px; font-size: 25px; line-height: 1.05; font-weight: 800; color: #0f172a; font-variant-numeric: tabular-nums; }}
    .leaderboard, .type-table, .case-list {{ width: 100%; min-width: 760px; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 9px 8px; text-align: left; vertical-align: top; }}
    th {{ color: #475569; font-weight: 800; background: #f8fafc; position: sticky; top: 0; z-index: 1; }}
    .metric {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .badge {{ display: inline-block; border-radius: 999px; padding: 3px 8px; background: #eef2ff; color: #3730a3; font-size: 12px; }}
    .our {{ background: #ecfdf5; color: #047857; }}
    .baseline {{ background: #f8fafc; color: #475569; }}
    .case-tabs {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; }}
    .case-tab {{ border: 1px solid #cbd5e1; background: white; border-radius: 999px; padding: 7px 12px; }}
    .case-tab.active {{ background: #111827; border-color: #111827; color: white; }}
    .player {{ margin-top: 0; }}
    .section-heading {{ display: flex; justify-content: space-between; gap: 14px; align-items: start; margin-bottom: 12px; }}
    .section-heading .subtle {{ max-width: 780px; text-align: right; font-variant-numeric: tabular-nums; }}
    .control-surface {{ display: grid; gap: 10px; margin-bottom: 12px; padding: 12px; border: 1px solid #e2e8f0; border-radius: 8px; background: #f8fafc; }}
    .player-tools {{ display: grid; grid-template-columns: auto minmax(220px, 1fr) auto; gap: 12px; align-items: end; }}
    .secondary-button {{ padding: 8px 14px; }}
    .secondary-button.active {{ background: #111827; border-color: #111827; color: white; }}
    .mode-switch {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
    .view-mode-button {{ min-height: 44px; padding: 7px 12px; }}
    .view-mode-button.active {{ background: #111827; border-color: #111827; color: white; }}
    .layer-switch {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
    .layer-button {{ min-height: 44px; padding: 7px 12px; }}
    .layer-button.active {{ background: #111827; border-color: #111827; color: white; }}
    .layer-switch[hidden], .comparison-grid[hidden], .single-view[hidden] {{ display: none; }}
    .heat-controls {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }}
    .heat-controls label {{ min-width: 180px; max-width: 260px; }}
    .sequence-stats {{ display: grid; grid-template-columns: repeat(4, minmax(130px, 1fr)); gap: 8px; }}
    .sequence-stat {{ border: 1px solid #e1e7ef; border-radius: 7px; padding: 8px 10px; background: white; }}
    .sequence-stat span {{ display: block; color: #64748b; font-size: 12px; }}
    .sequence-stat strong {{ display: block; margin-top: 3px; font-size: 17px; }}
    #frameBadge {{ color: #475569; font-size: 13px; font-variant-numeric: tabular-nums; }}
    .canvas-shell {{ background: #111827; border-radius: 8px; padding: 10px; }}
    .comparison-grid {{ display: grid; grid-template-columns: repeat(2, minmax(260px, 1fr)); gap: 12px; }}
    .video-panel {{ min-width: 0; margin: 0; border: 1px solid #1f2937; border-radius: 8px; padding: 9px; background: #0f172a; }}
    .video-panel figcaption {{ display: flex; align-items: center; min-height: 24px; margin: 0 0 7px; color: #f8fafc; font-size: 12px; font-weight: 800; }}
    canvas {{ display: block; width: 100%; height: auto; background: #e2e8f0; border-radius: 6px; }}
    section {{ margin-top: 16px; }}
    .analysis-tabs {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }}
    .analysis-tab.active {{ background: #111827; border-color: #111827; color: white; }}
    .analysis-panel-block[hidden] {{ display: none; }}
    .table-scroll {{ overflow-x: auto; -webkit-overflow-scrolling: touch; border: 1px solid #eef2f7; border-radius: 8px; }}
    .table-scroll table {{ background: white; }}
    @media (prefers-reduced-motion: reduce) {{
      button {{ transition: none; }}
    }}
    @media (max-width: 960px) {{
      main {{ padding: 16px; }}
      header {{ display: grid; }}
      .cards {{ grid-template-columns: repeat(2, minmax(120px, 1fr)); }}
      select {{ min-width: 0; width: 100%; }}
      .player-tools {{ grid-template-columns: 1fr; }}
      .heat-controls label {{ max-width: none; width: 100%; }}
      .sequence-stats {{ grid-template-columns: repeat(2, minmax(120px, 1fr)); }}
      .comparison-grid {{ grid-template-columns: 1fr; }}
      .toolbar {{ display: grid; width: 100%; justify-content: stretch; }}
      .toolbar label {{ width: 100%; }}
      .section-heading {{ display: grid; }}
      .section-heading .subtle {{ text-align: left; }}
      .control-surface {{ padding: 10px; }}
      .mode-switch button, .layer-switch button {{ flex: 1 1 140px; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1 data-i18n="title">FusionTrack 最终结果看板</h1>
        <div class="subtle" data-i18n="subtitle">多方法多模态异常检测实验展示</div>
      </div>
      <div class="toolbar">
        <label><span data-i18n="language">语言</span>
          <select id="languageSelector">
            <option value="zh">中文</option>
            <option value="en">English</option>
          </select>
        </label>
        <label><span data-i18n="task">任务</span>
          <select id="taskSelector"></select>
        </label>
        <label><span data-i18n="method">方法</span>
          <select id="methodSelector"></select>
        </label>
        <label><span data-i18n="sequence">序列</span>
          <select id="sequenceSelector"></select>
        </label>
      </div>
    </header>
    <div id="cards" class="cards"></div>

    <section class="panel player">
      <div class="section-heading">
        <h2 data-i18n="interactivePlayback">Interactive playback</h2>
        <div class="subtle" id="playbackReadout">No playback loaded</div>
      </div>
      <div class="control-surface">
        <div class="player-tools">
          <button type="button" class="secondary-button" id="playToggle">Play</button>
          <label><span data-i18n="frame">帧</span>
            <input id="frameSlider" type="range" min="0" max="0" value="0">
          </label>
          <span id="frameBadge">0 / 0</span>
        </div>
        <div class="mode-switch" aria-label="Visualization mode">
          <button type="button" class="view-mode-button active" data-view-mode="comparison" data-i18n="viewComparison">四画面对比</button>
          <button type="button" class="view-mode-button" data-view-mode="single" data-i18n="viewSingle">单画面模式</button>
        </div>
        <div class="layer-switch" id="singleLayerSwitch" aria-label="Single playback layer" hidden>
          <span class="subtle" data-i18n="singleLayerLabel">单画面图层</span>
          <button type="button" class="layer-button" data-layer="tracks" data-i18n="layerTracks">Tracks</button>
          <button type="button" class="layer-button active" data-layer="both" data-i18n="layerBoth">Heat + Tracks</button>
          <button type="button" class="layer-button" data-layer="heatmap" data-i18n="layerHeatmap">Heatmap</button>
        </div>
        <div class="heat-controls">
          <label><span data-i18n="heatOpacityLabel">热力透明度</span>
            <input id="heatOpacity" type="range" min="0" max="100" value="64">
          </label>
          <label><span data-i18n="timeWindowLabel">时间窗口</span>
            <input id="heatWindow" type="range" min="12" max="120" value="36">
          </label>
        </div>
        <div id="sequenceStats" class="sequence-stats"></div>
      </div>
      <div id="comparisonView" class="comparison-grid">
        <figure class="video-panel">
          <figcaption data-i18n="panelOriginal">原视频</figcaption>
          <canvas id="originalCanvas" width="960" height="612"></canvas>
        </figure>
        <figure class="video-panel">
          <figcaption data-i18n="panelHeatmap">热力图</figcaption>
          <canvas id="heatmapCanvas" width="960" height="612"></canvas>
        </figure>
        <figure class="video-panel">
          <figcaption data-i18n="panelTracks">轨迹</figcaption>
          <canvas id="tracksCanvas" width="960" height="612"></canvas>
        </figure>
        <figure class="video-panel">
          <figcaption data-i18n="panelBoth">热力 + 轨迹</figcaption>
          <canvas id="bothCanvas" width="960" height="612"></canvas>
        </figure>
      </div>
      <div id="singleView" class="single-view" hidden>
        <div class="canvas-shell"><canvas id="singleCanvas" width="960" height="612"></canvas></div>
      </div>
    </section>

    <section class="panel">
      <div class="section-heading">
        <h2 data-i18n="analysisTitle">实验分析</h2>
      </div>
      <div class="analysis-tabs">
        <button type="button" class="analysis-tab active" data-panel="leaderboard" data-i18n="tabLeaderboard">方法排名</button>
        <button type="button" class="analysis-tab" data-panel="types" data-i18n="tabTypes">异常类型分析</button>
        <button type="button" class="analysis-tab" data-panel="cases" data-i18n="tabCases">典型案例</button>
      </div>
      <div class="analysis-panel-block" data-analysis-panel="leaderboard">
        <div class="table-scroll"><table class="leaderboard" id="leaderboardTable"></table></div>
      </div>
      <div class="analysis-panel-block" data-analysis-panel="types" hidden>
        <div class="table-scroll"><table class="type-table" id="typeTable"></table></div>
      </div>
      <div class="analysis-panel-block" data-analysis-panel="cases" hidden>
        <div class="case-tabs">
          <button type="button" class="case-tab active" data-case="true_positive">True Positive</button>
          <button type="button" class="case-tab" data-case="false_positive">False Positive</button>
          <button type="button" class="case-tab" data-case="false_negative">False Negative</button>
        </div>
        <div class="table-scroll"><table class="case-list" id="caseTable"></table></div>
      </div>
    </section>
  </main>
  <script id="dashboardData" type="application/json">{dashboard_json}</script>
  <script id="playbackData" type="application/json">{playback_json}</script>
  <script>
    (() => {{
      const dashboard = JSON.parse(document.getElementById("dashboardData").textContent);
      const playbackData = JSON.parse(document.getElementById("playbackData").textContent);
      const languageSelector = document.getElementById("languageSelector");
      const taskSelector = document.getElementById("taskSelector");
      const methodSelector = document.getElementById("methodSelector");
      const cards = document.getElementById("cards");
      const leaderboardTable = document.getElementById("leaderboardTable");
      const typeTable = document.getElementById("typeTable");
      const caseTable = document.getElementById("caseTable");
      const caseTabs = Array.from(document.querySelectorAll(".case-tab"));
      const analysisTabs = Array.from(document.querySelectorAll(".analysis-tab"));
      const analysisPanels = Array.from(document.querySelectorAll("[data-analysis-panel]"));
      const canvases = {{
        original: document.getElementById("originalCanvas"),
        heatmap: document.getElementById("heatmapCanvas"),
        tracks: document.getElementById("tracksCanvas"),
        both: document.getElementById("bothCanvas"),
        single: document.getElementById("singleCanvas")
      }};
      const comparisonView = document.getElementById("comparisonView");
      const singleView = document.getElementById("singleView");
      const singleLayerSwitch = document.getElementById("singleLayerSwitch");
      const playbackReadout = document.getElementById("playbackReadout");
      const sequenceSelector = document.getElementById("sequenceSelector");
      const playToggle = document.getElementById("playToggle");
      const frameSlider = document.getElementById("frameSlider");
      const frameBadge = document.getElementById("frameBadge");
      const heatOpacity = document.getElementById("heatOpacity");
      const heatWindow = document.getElementById("heatWindow");
      const sequenceStats = document.getElementById("sequenceStats");
      const viewModeButtons = Array.from(document.querySelectorAll(".view-mode-button"));
      const layerButtons = Array.from(document.querySelectorAll(".layer-button"));
      const translations = {{
        zh: {{
          documentTitle: "FusionTrack 最终结果看板",
          title: "FusionTrack 最终结果看板",
          subtitle: "多方法多模态异常检测实验展示",
          language: "语言",
          task: "任务",
          method: "方法",
          sequence: "序列",
          cardMethods: "方法数",
          cardLabels: "总标签数",
          cardPositives: "总异常数",
          cardAuroc: "当前 AUROC",
          sequenceSampleCount: "当前序列样本数",
          sequenceAnomalyCount: "当前序列异常数",
          sequenceFrameRange: "当前序列帧范围",
          sequenceVisualizedTracks: "可视化轨迹数",
          analysisTitle: "实验分析",
          tabLeaderboard: "方法排名",
          tabTypes: "异常类型分析",
          tabCases: "典型案例",
          interactivePlayback: "动态可视化",
          frame: "帧",
          play: "播放",
          pause: "暂停",
          viewComparison: "四画面对比",
          viewSingle: "单画面模式",
          singleLayerLabel: "单画面图层",
          panelOriginal: "原视频",
          panelHeatmap: "热力图",
          panelTracks: "轨迹",
          panelBoth: "热力 + 轨迹",
          layerTracks: "轨迹",
          layerBoth: "热力 + 轨迹",
          layerHeatmap: "热力图",
          heatOpacityLabel: "热力透明度",
          timeWindowLabel: "时间窗口",
          noPlayback: "当前任务没有可播放轨迹。",
          playbackPrefix: "可视化",
          visibleTracks: "条轨迹",
          methodHeader: "方法",
          roleHeader: "角色",
          anomalyTypeHeader: "异常类型",
          hitsHeader: "命中@K",
          totalHeader: "总数",
          recallHeader: "召回@K",
          meanScoreHeader: "平均正样本分数",
          sampleHeader: "样本",
          typeHeader: "类型",
          scoreHeader: "分数",
          rankHeader: "排名",
          framesHeader: "帧范围",
          truePositive: "正确检出",
          falsePositive: "误报",
          falseNegative: "漏报",
          taskIndividual: "Individual",
          taskGroup: "Group",
          view_comparison: "四画面对比",
          view_single: "单画面模式",
          layer_tracks: "轨迹",
          layer_both: "热力 + 轨迹",
          layer_heatmap: "热力图"
        }},
        en: {{
          documentTitle: "FusionTrack Final Results Dashboard",
          title: "Final Results Dashboard",
          subtitle: "Multi-method FusionTrack anomaly benchmark",
          language: "Language",
          task: "Task",
          method: "Method",
          sequence: "Sequence",
          cardMethods: "Methods",
          cardLabels: "Total labels",
          cardPositives: "Total anomalies",
          cardAuroc: "Selected AUROC",
          sequenceSampleCount: "Sequence samples",
          sequenceAnomalyCount: "Sequence anomalies",
          sequenceFrameRange: "Sequence frame range",
          sequenceVisualizedTracks: "Visualized tracks",
          analysisTitle: "Experiment Analysis",
          tabLeaderboard: "Method Ranking",
          tabTypes: "Anomaly-Type Analysis",
          tabCases: "Representative Cases",
          interactivePlayback: "Interactive Playback",
          frame: "Frame",
          play: "Play",
          pause: "Pause",
          viewComparison: "Four-panel comparison",
          viewSingle: "Single view",
          singleLayerLabel: "Single-view layer",
          panelOriginal: "Original",
          panelHeatmap: "Heatmap",
          panelTracks: "Tracks",
          panelBoth: "Heat + Tracks",
          layerTracks: "Tracks",
          layerBoth: "Heat + Tracks",
          layerHeatmap: "Heatmap",
          heatOpacityLabel: "Heat opacity",
          timeWindowLabel: "Time window",
          noPlayback: "Playback is not available for the current task.",
          playbackPrefix: "Playback",
          visibleTracks: "visible tracks",
          methodHeader: "Method",
          roleHeader: "Role",
          anomalyTypeHeader: "Anomaly type",
          hitsHeader: "Hits@K",
          totalHeader: "Total",
          recallHeader: "Recall@K",
          meanScoreHeader: "Mean positive score",
          sampleHeader: "Sample",
          typeHeader: "Type",
          scoreHeader: "Score",
          rankHeader: "Rank",
          framesHeader: "Frames",
          truePositive: "True Positive",
          falsePositive: "False Positive",
          falseNegative: "False Negative",
          taskIndividual: "Individual",
          taskGroup: "Group",
          view_comparison: "four-panel comparison",
          view_single: "single view",
          layer_tracks: "tracks",
          layer_both: "heat + tracks",
          layer_heatmap: "heatmap"
        }}
      }};
      const backgroundCache = new Map();
      const state = {{
        language: localStorage.getItem("fusiontrack.finalDashboard.language") || "zh",
        task: "{html.escape(initial_task)}",
        method: "{html.escape(initial_method)}",
        caseType: "true_positive",
        sequence: "{html.escape(initial_sequence)}",
        frame: -1,
        playing: false,
        viewMode: "comparison",
        layer: "both",
        heatOpacity: 0.64,
        heatWindow: 36,
        image: null,
        imageKey: null,
        timer: null
      }};

      function taskData() {{ return dashboard.tasks[state.task]; }}
      function fmt(value) {{ return Number(value || 0).toFixed(3); }}
      function esc(value) {{ return String(value ?? "").replace(/[&<>"']/g, ch => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}})[ch]); }}
      function methodsForTask(task) {{ return task.leaderboard.map(row => row.method); }}
      function sequences() {{ return Object.keys(playbackData); }}
      function currentPlayback() {{ return playbackData[state.sequence] || playbackData[sequences()[0]] || null; }}
      function clamp(value, min, max) {{ return Math.max(min, Math.min(max, value)); }}
      function t(key) {{ return (translations[state.language] || translations.zh)[key] || translations.en[key] || key; }}

      function setTaskOptions() {{
        const labels = {{ individual: t("taskIndividual"), group: t("taskGroup") }};
        taskSelector.innerHTML = Object.keys(dashboard.tasks).map(task =>
          `<option value="${{esc(task)}}"${{task === state.task ? " selected" : ""}}>${{esc(labels[task] || task)}}</option>`
        ).join("");
      }}

      function applyLanguage(language) {{
        state.language = translations[language] ? language : "zh";
        languageSelector.value = state.language;
        document.documentElement.lang = state.language === "zh" ? "zh-CN" : "en";
        document.title = t("documentTitle");
        localStorage.setItem("fusiontrack.finalDashboard.language", state.language);
        document.querySelectorAll("[data-i18n]").forEach(element => {{
          element.textContent = t(element.dataset.i18n);
        }});
        setTaskOptions();
      }}

      function setMethodOptions() {{
        const task = taskData();
        methodSelector.innerHTML = methodsForTask(task).map(method =>
          `<option value="${{esc(method)}}"${{method === state.method ? " selected" : ""}}>${{esc(method)}}</option>`
        ).join("");
      }}

      function setSequenceOptions() {{
        const names = sequences();
        if (!names.length) {{
          sequenceSelector.innerHTML = "";
          sequenceSelector.disabled = true;
          playToggle.disabled = true;
          frameSlider.disabled = true;
          return;
        }}
        if (!state.sequence || !playbackData[state.sequence]) {{
          state.sequence = names[0];
        }}
        sequenceSelector.disabled = state.task !== "individual";
        playToggle.disabled = state.task !== "individual";
        frameSlider.disabled = state.task !== "individual";
        sequenceSelector.innerHTML = names.map(sequence =>
          `<option value="${{esc(sequence)}}"${{sequence === state.sequence ? " selected" : ""}}>${{esc(sequence)}}</option>`
        ).join("");
      }}

      function renderCards() {{
        const task = taskData();
        const current = task.methods[state.method];
        const metrics = current.metrics;
        cards.innerHTML = [
          [t("cardMethods"), Object.keys(task.methods).length],
          [t("cardLabels"), task.num_labels],
          [t("cardPositives"), task.num_positive],
          [t("cardAuroc"), fmt(metrics.auroc)]
        ].map(([label, value]) => `<div class="card"><div>${{label}}</div><div class="value">${{value}}</div></div>`).join("");
      }}

      function renderSequenceStats() {{
        const data = currentPlayback();
        if (!data || state.task !== "individual") {{
          sequenceStats.innerHTML = "";
          return;
        }}
        const stats = data.stats || {{}};
        const frameStart = stats.frame_start ?? data.frame_range?.[0] ?? 0;
        const frameEnd = stats.frame_end ?? data.frame_range?.[1] ?? frameStart;
        const rows = [
          [t("sequenceSampleCount"), stats.sequence_sample_count ?? data.tracks.length],
          [t("sequenceAnomalyCount"), stats.sequence_anomaly_count ?? data.tracks.filter(track => track.label === 1).length],
          [t("sequenceFrameRange"), `${{frameStart}}-${{frameEnd}}`],
          [t("sequenceVisualizedTracks"), stats.visualized_tracks ?? data.tracks.length]
        ];
        sequenceStats.innerHTML = rows.map(([label, value]) => `
          <div class="sequence-stat"><span>${{label}}</span><strong>${{value}}</strong></div>
        `).join("");
      }}

      function renderLeaderboard() {{
        const rows = taskData().leaderboard.slice(0, 8);
        leaderboardTable.innerHTML = `
          <thead><tr><th>${{t("methodHeader")}}</th><th>${{t("roleHeader")}}</th><th class="metric">AUROC</th><th class="metric">AUPRC</th><th class="metric">F1</th><th class="metric">P@100</th><th class="metric">R@100</th></tr></thead>
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
          <thead><tr><th>${{t("anomalyTypeHeader")}}</th><th class="metric">${{t("hitsHeader")}}</th><th class="metric">${{t("totalHeader")}}</th><th class="metric">${{t("recallHeader")}}</th><th class="metric">${{t("meanScoreHeader")}}</th></tr></thead>
          <tbody>${{rows.map(row => `
            <tr><td>${{esc(row.anomaly_type)}}</td><td class="metric">${{row.hits_at_k}}</td><td class="metric">${{row.total_positive}}</td><td class="metric">${{fmt(row.recall_at_k)}}</td><td class="metric">${{fmt(row.mean_positive_score)}}</td></tr>
          `).join("")}}</tbody>
        `;
      }}

      function renderCases() {{
        const rows = ((taskData().case_rows[state.method] || {{}})[state.caseType] || []);
        caseTable.innerHTML = `
          <thead><tr><th>${{t("sampleHeader")}}</th><th>${{t("typeHeader")}}</th><th class="metric">${{t("scoreHeader")}}</th><th class="metric">${{t("rankHeader")}}</th><th>${{t("framesHeader")}}</th></tr></thead>
          <tbody>${{rows.map(row => `
            <tr><td><strong>${{esc(row.sequence)}} / ${{esc(row.track_id)}}</strong><br><span class="subtle">${{esc(row.sample_id)}}</span></td><td>${{esc(row.anomaly_type)}}</td><td class="metric">${{fmt(row.score)}}</td><td class="metric">${{row.rank}}</td><td>${{row.frame_start}}-${{row.frame_end}}</td></tr>
          `).join("")}}</tbody>
        `;
      }}

      function renderCaseTabs() {{
        const labels = {{
          true_positive: t("truePositive"),
          false_positive: t("falsePositive"),
          false_negative: t("falseNegative")
        }};
        caseTabs.forEach(button => {{
          button.textContent = labels[button.dataset.case] || button.dataset.case;
        }});
      }}

      function setAnalysisPanel(panel) {{
        analysisTabs.forEach(tab => tab.classList.toggle("active", tab.dataset.panel === panel));
        analysisPanels.forEach(item => {{
          item.hidden = item.dataset.analysisPanel !== panel;
        }});
      }}

      function resetFrameForSequence() {{
        const data = currentPlayback();
        if (!data) {{
          state.frame = 0;
          frameSlider.min = 0;
          frameSlider.max = 0;
          frameSlider.value = 0;
          frameBadge.textContent = "0 / 0";
          return;
        }}
        const start = Number(data.frame_range?.[0] || 0);
        const end = Number(data.frame_range?.[1] || start);
        if (state.frame < start || state.frame > end) {{
          state.frame = Math.round(start + (end - start) * 0.35);
        }}
        frameSlider.min = start;
        frameSlider.max = end;
        frameSlider.value = state.frame;
        frameBadge.textContent = `${{state.frame}} / ${{end}}`;
      }}

      function backgroundForFrame(data, frame) {{
        const frames = data.background_frames || [];
        if (!frames.length) {{
          return data.background ? {{ frame: data.frame_range?.[0] || 0, src: data.background }} : null;
        }}
        let selected = frames[0];
        for (const item of frames) {{
          if (Number(item.frame) <= frame) {{
            selected = item;
          }} else {{
            break;
          }}
        }}
        return selected;
      }}

      function ensureBackground(data, frame) {{
        const background = backgroundForFrame(data, frame);
        if (!background || !background.src) {{
          state.image = null;
          state.imageKey = null;
          return;
        }}
        const key = `${{data.sequence}}:${{background.src}}`;
        if (state.imageKey === key) {{
          return;
        }}
        state.imageKey = key;
        if (backgroundCache.has(key)) {{
          state.image = backgroundCache.get(key);
          return;
        }}
        const image = new Image();
        image.onload = () => {{
          backgroundCache.set(key, image);
          if (state.imageKey === key) {{
            state.image = image;
            drawPlayback();
          }}
        }};
        image.onerror = () => {{
          if (state.imageKey === key) {{
            state.image = null;
            drawPlayback();
          }}
        }};
        image.src = background.src;
      }}

      function activePoints(track, frame) {{
        const points = track.points.filter(point => Number(point.frame) <= frame);
        return points.length ? points : track.points.slice(0, 1);
      }}

      function heatPoints(track, frame) {{
        const start = frame - state.heatWindow;
        const points = track.points.filter(point => Number(point.frame) <= frame && Number(point.frame) >= start);
        const visible = points.length ? points : activePoints(track, frame).slice(-1);
        if (visible.length <= 12) {{
          return visible;
        }}
        const stride = Math.ceil(visible.length / 12);
        return visible.filter((_, index) => index % stride === 0).slice(-12);
      }}

      function setCanvasSize(targetCanvas, data) {{
        const width = Number(data.size?.width || 960);
        const height = Number(data.size?.height || 612);
        if (targetCanvas.width !== width) {{
          targetCanvas.width = width;
        }}
        if (targetCanvas.height !== height) {{
          targetCanvas.height = height;
        }}
      }}

      function clearPlaybackCanvases() {{
        Object.values(canvases).forEach(targetCanvas => {{
          if (!targetCanvas) {{
            return;
          }}
          const targetCtx = targetCanvas.getContext("2d");
          targetCtx.clearRect(0, 0, targetCanvas.width, targetCanvas.height);
        }});
      }}

      function setViewModeVisibility() {{
        const comparison = state.viewMode === "comparison";
        comparisonView.hidden = !comparison;
        singleView.hidden = comparison;
        singleLayerSwitch.hidden = comparison;
        viewModeButtons.forEach(button => {{
          button.classList.toggle("active", button.dataset.viewMode === state.viewMode);
        }});
      }}

      function drawCanvasBase(targetCtx, targetCanvas, data, layer) {{
        targetCtx.fillStyle = "#e2e8f0";
        targetCtx.fillRect(0, 0, targetCanvas.width, targetCanvas.height);
        if (state.image) {{
          targetCtx.drawImage(state.image, 0, 0, targetCanvas.width, targetCanvas.height);
        }}
        if (layer === "heatmap" || layer === "both") {{
          targetCtx.save();
          targetCtx.fillStyle = layer === "heatmap" ? "rgba(4, 9, 18, 0.18)" : "rgba(4, 9, 18, 0.06)";
          targetCtx.fillRect(0, 0, targetCanvas.width, targetCanvas.height);
          targetCtx.restore();
        }}
      }}

      function drawHeatmap(targetCtx, targetCanvas, data, ranked, maxScore) {{
        const heatCanvas = document.createElement("canvas");
        heatCanvas.width = targetCanvas.width;
        heatCanvas.height = targetCanvas.height;
        const heatCtx = heatCanvas.getContext("2d");
        heatCtx.clearRect(0, 0, heatCanvas.width, heatCanvas.height);
        heatCtx.globalCompositeOperation = "lighter";
        const currentFrame = Number(state.frame);
        for (const track of ranked.slice(0, 55)) {{
          const score = Number((track.method_scores || {{}})[state.method] || 0);
          const scoreRatio = clamp(score / maxScore, 0, 1);
          const labelBoost = track.label === 1 ? 0.22 : 0;
          for (const point of heatPoints(track, currentFrame)) {{
            const age = Math.max(0, currentFrame - Number(point.frame));
            const recency = clamp(1 - age / Math.max(1, state.heatWindow), 0.18, 1);
            const strength = clamp((0.16 + 0.72 * scoreRatio + labelBoost) * recency, 0.08, 1);
            const radius = 14 + 26 * strength;
            const gradient = heatCtx.createRadialGradient(point.x, point.y, 0, point.x, point.y, radius);
            gradient.addColorStop(0, `rgba(255, 43, 85, ${{0.30 * strength}})`);
            gradient.addColorStop(0.28, `rgba(255, 179, 0, ${{0.23 * strength}})`);
            gradient.addColorStop(0.62, `rgba(45, 212, 191, ${{0.15 * strength}})`);
            gradient.addColorStop(1, "rgba(45, 212, 191, 0)");
            heatCtx.fillStyle = gradient;
            heatCtx.beginPath();
            heatCtx.arc(point.x, point.y, radius, 0, Math.PI * 2);
            heatCtx.fill();
          }}
        }}
        targetCtx.save();
        targetCtx.globalAlpha = state.heatOpacity;
        targetCtx.globalCompositeOperation = "screen";
        targetCtx.drawImage(heatCanvas, 0, 0);
        targetCtx.restore();
      }}

      function drawTracks(targetCtx, ranked, maxScore) {{
        for (const track of ranked) {{
          const points = activePoints(track, state.frame);
          if (!points.length) {{
            continue;
          }}
          const score = Number((track.method_scores || {{}})[state.method] || 0);
          const ratio = clamp(score / maxScore, 0, 1);
          targetCtx.strokeStyle = track.label === 1 ? "rgba(239, 68, 68, 0.95)" : `rgba(37, 99, 235, ${{0.25 + 0.45 * ratio}})`;
          targetCtx.lineWidth = track.label === 1 ? 2.6 : 0.9 + 2.0 * ratio;
          targetCtx.beginPath();
          points.forEach((point, index) => index ? targetCtx.lineTo(point.x, point.y) : targetCtx.moveTo(point.x, point.y));
          targetCtx.stroke();
          const last = points[points.length - 1];
          if (last) {{
            targetCtx.fillStyle = track.label === 1 ? "#ef4444" : "#f59e0b";
            targetCtx.beginPath();
            targetCtx.arc(last.x, last.y, track.label === 1 ? 4.8 : 2.7, 0, Math.PI * 2);
            targetCtx.fill();
          }}
        }}
      }}

      function drawCanvasLayer(targetCanvas, layer, data, ranked, maxScore) {{
        if (!targetCanvas) {{
          return;
        }}
        setCanvasSize(targetCanvas, data);
        const targetCtx = targetCanvas.getContext("2d");
        drawCanvasBase(targetCtx, targetCanvas, data, layer);
        if (layer === "original") {{
          return;
        }}
        if (layer === "heatmap" || layer === "both") {{
          drawHeatmap(targetCtx, targetCanvas, data, ranked, maxScore);
        }}
        if (layer === "tracks" || layer === "both") {{
          drawTracks(targetCtx, ranked, maxScore);
        }}
      }}

      function drawComparisonView(data, ranked, maxScore) {{
        drawCanvasLayer(canvases.original, "original", data, ranked, maxScore);
        drawCanvasLayer(canvases.heatmap, "heatmap", data, ranked, maxScore);
        drawCanvasLayer(canvases.tracks, "tracks", data, ranked, maxScore);
        drawCanvasLayer(canvases.both, "both", data, ranked, maxScore);
      }}

      function drawSingleView(data, ranked, maxScore) {{
        drawCanvasLayer(canvases.single, state.layer, data, ranked, maxScore);
      }}

      function drawPlayback() {{
        setViewModeVisibility();
        const names = sequences();
        if (!names.length || state.task !== "individual") {{
          playbackReadout.textContent = t("noPlayback");
          clearPlaybackCanvases();
          return;
        }}
        const data = currentPlayback();
        if (!data) {{
          return;
        }}
        resetFrameForSequence();
        ensureBackground(data, state.frame);
        const scores = data.tracks.map(track => Number((track.method_scores || {{}})[state.method] || 0));
        const maxScore = Math.max(...scores, 1e-6);
        const ranked = [...data.tracks].sort((a, b) => Number((b.method_scores || {{}})[state.method] || 0) - Number((a.method_scores || {{}})[state.method] || 0)).slice(0, 80);
        if (state.viewMode === "comparison") {{
          drawComparisonView(data, ranked, maxScore);
        }} else {{
          drawSingleView(data, ranked, maxScore);
        }}
        const viewLabel = state.viewMode === "comparison" ? t("view_comparison") : `${{t("view_single")}} - ${{t(`layer_${{state.layer}}`)}}`;
        playbackReadout.textContent = `${{t("playbackPrefix")}} / ${{data.sequence}} / ${{state.method}} / ${{t("frame")}} ${{state.frame}} / ${{viewLabel}} / ${{ranked.length}} ${{t("visibleTracks")}}`;
      }}

      function stopPlayback() {{
        state.playing = false;
        playToggle.textContent = t("play");
        playToggle.classList.remove("active");
        if (state.timer) {{
          window.clearInterval(state.timer);
          state.timer = null;
        }}
      }}

      function startPlayback() {{
        if (state.task !== "individual" || !currentPlayback()) {{
          return;
        }}
        state.playing = true;
        playToggle.textContent = t("pause");
        playToggle.classList.add("active");
        state.timer = window.setInterval(() => {{
          const data = currentPlayback();
          const start = Number(data.frame_range?.[0] || 0);
          const end = Number(data.frame_range?.[1] || start);
          state.frame = state.frame >= end ? start : state.frame + 1;
          frameSlider.value = state.frame;
          drawPlayback();
        }}, 90);
      }}

      function renderMethodView() {{
        setTaskOptions();
        setMethodOptions();
        setSequenceOptions();
        playToggle.textContent = state.playing ? t("pause") : t("play");
        renderCards();
        renderSequenceStats();
        renderCaseTabs();
        renderLeaderboard();
        renderTypeTable();
        renderCases();
        drawPlayback();
      }}

      languageSelector.addEventListener("change", () => {{
        applyLanguage(languageSelector.value);
        renderMethodView();
      }});
      taskSelector.addEventListener("change", () => {{
        state.task = taskSelector.value;
        state.method = methodsForTask(taskData())[0] || "";
        if (state.task !== "individual") {{
          stopPlayback();
        }}
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
      analysisTabs.forEach(button => button.addEventListener("click", () => {{
        setAnalysisPanel(button.dataset.panel);
      }}));
      sequenceSelector.addEventListener("change", () => {{
        state.sequence = sequenceSelector.value;
        state.image = null;
        state.imageKey = null;
        state.frame = -1;
        resetFrameForSequence();
        renderSequenceStats();
        drawPlayback();
      }});
      frameSlider.addEventListener("input", () => {{
        state.frame = Number(frameSlider.value);
        drawPlayback();
      }});
      heatOpacity.addEventListener("input", () => {{
        state.heatOpacity = Number(heatOpacity.value) / 100;
        drawPlayback();
      }});
      heatWindow.addEventListener("input", () => {{
        state.heatWindow = Number(heatWindow.value);
        drawPlayback();
      }});
      viewModeButtons.forEach(button => button.addEventListener("click", () => {{
        state.viewMode = button.dataset.viewMode || "comparison";
        setViewModeVisibility();
        drawPlayback();
      }}));
      layerButtons.forEach(button => button.addEventListener("click", () => {{
        state.layer = button.dataset.layer;
        layerButtons.forEach(item => item.classList.toggle("active", item === button));
        drawPlayback();
      }}));
      playToggle.addEventListener("click", () => {{
        state.playing ? stopPlayback() : startPlayback();
      }});
      applyLanguage(state.language);
      setAnalysisPanel("leaderboard");
      renderMethodView();
    }})();
  </script>
</body>
</html>
"""
