"""Responsive AgenticDerma platform with live process visibility."""

from __future__ import annotations

import base64
import binascii
import json
import re
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "main"))
import derma_agent as project  # noqa: E402
from agentic_derma import AgenticDerma  # noqa: E402


OUTPUT_ROOT = ROOT / "output" / "platform"
DEFAULT_IMAGE = ROOT / "input" / "sample_derma.png"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
ALLOWED_FILES = {"attribution.png", "prediction.json", "confusion_matrix.png", "calibration_plot.png"}
JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
MODEL_LOCK = threading.Lock()
COORDINATOR = AgenticDerma()


PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
  <title>AgenticDerma</title>
  <style>
    :root{
      --ink:#1a1c1f;--ink-soft:#4d5157;--muted:#8a8d93;--line:#e6e4e0;--line-soft:#efeee9;
      --surface:#ffffff;--canvas:#f6f5f1;--canvas-alt:#efeee7;
      --accent:#4338ca;--accent-strong:#332c9e;--accent-soft:#edecfb;
      --good:#0f8a5f;--good-soft:#e5f5ee;--warn:#b5760f;--warn-soft:#fbf1de;--danger:#c23b52;--danger-soft:#fbeaed;
      --radius-sm:8px;--radius-md:13px;--radius-lg:20px;--radius-pill:999px;
      --shadow-sm:0 1px 2px rgba(23,24,28,.05),0 1px 1px rgba(23,24,28,.04);
      --shadow-md:0 10px 30px rgba(23,24,28,.08);
      --shadow-lg:0 28px 70px rgba(23,24,28,.20);
      --ease:cubic-bezier(.2,.7,.3,1);
    }
    *{box-sizing:border-box}
    html,body{height:100%;margin:0}
    body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,ui-sans-serif,system-ui,sans-serif;color:var(--ink);background:var(--canvas);overflow:hidden;-webkit-font-smoothing:antialiased}
    button,input,textarea,select{font:inherit;color:inherit}
    button{cursor:pointer;background:none}
    button:disabled{cursor:not-allowed}
    svg{display:block}
    .icon{width:18px;height:18px;stroke:currentColor;stroke-width:1.8;fill:none;stroke-linecap:round;stroke-linejoin:round;flex:none}

    /* ---- shell & topbar ---- */
    .shell{height:100%;display:grid;grid-template-rows:auto minmax(0,1fr)}
    .topbar{display:flex;align-items:center;gap:10px;padding:12px 20px;border-bottom:1px solid var(--line);background:var(--surface);position:relative;z-index:20}
    .brand{display:flex;align-items:center;gap:10px;margin-right:auto}
    .brand .mark{width:32px;height:32px;border-radius:9px;background:var(--ink);color:#fff;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:12px;letter-spacing:.02em}
    .brand strong{font-size:14px;font-weight:650;letter-spacing:-.01em}
    .topbar-actions{display:flex;align-items:center;gap:6px}

    .status-btn{display:flex;align-items:center;gap:7px;height:34px;padding:0 12px 0 10px;border-radius:var(--radius-pill);border:1px solid var(--line);background:var(--surface);font-size:12.5px;font-weight:560;transition:border-color .15s var(--ease),background .15s var(--ease)}
    .status-btn:hover{border-color:#d5d3cd;background:var(--canvas-alt)}
    .status-dot{width:7px;height:7px;border-radius:50%;background:var(--muted);flex:none;transition:background .2s}
    .status-dot.good{background:var(--good)}
    .status-dot.warn{background:var(--warn);animation:blink 1.6s ease-in-out infinite}
    .status-dot.busy{background:var(--accent);animation:blink 1s ease-in-out infinite}
    @keyframes blink{50%{opacity:.35}}
    .status-btn .chev{width:12px;height:12px;stroke-width:2;transition:transform .18s var(--ease)}
    .status-btn[aria-expanded="true"] .chev{transform:rotate(180deg)}

    .icon-btn{display:inline-flex;align-items:center;justify-content:center;width:34px;height:34px;border-radius:var(--radius-pill);border:1px solid var(--line);background:var(--surface);transition:border-color .15s var(--ease),background .15s var(--ease)}
    .icon-btn:hover{border-color:#d5d3cd;background:var(--canvas-alt)}
    .icon-btn.subtle{border-color:transparent;background:transparent}
    .icon-btn.subtle:hover{background:var(--canvas-alt);border-color:var(--line)}
    .text-btn{display:inline-flex;align-items:center;gap:7px;height:34px;padding:0 13px;border-radius:var(--radius-pill);border:1px solid var(--line);background:var(--surface);font-size:12.5px;font-weight:600;transition:border-color .15s var(--ease),background .15s var(--ease)}
    .text-btn:hover{border-color:#d5d3cd;background:var(--canvas-alt)}
    .text-btn.running{position:relative}
    .text-btn.running:after{content:"";position:absolute;top:-2px;right:-2px;width:8px;height:8px;border-radius:50%;background:var(--accent);box-shadow:0 0 0 3px var(--accent-soft)}

    /* ---- status popover ---- */
    .popover{position:absolute;top:52px;right:20px;width:270px;background:var(--surface);border:1px solid var(--line);border-radius:var(--radius-md);box-shadow:var(--shadow-lg);padding:6px;opacity:0;transform:translateY(-6px) scale(.98);pointer-events:none;transition:opacity .15s var(--ease),transform .15s var(--ease);z-index:60}
    .popover.open{opacity:1;transform:none;pointer-events:auto}
    .popover-row{display:flex;align-items:center;gap:10px;padding:9px 10px;border-radius:var(--radius-sm)}
    .popover-row .status-dot{width:7px;height:7px}
    .popover-row .label{flex:1;font-size:12.5px;font-weight:550}
    .popover-row .value{font-size:11.5px;color:var(--muted);text-align:right}
    .popover-divider{height:1px;background:var(--line-soft);margin:4px 6px}

    /* ---- messages ---- */
    .stage-area{position:relative;min-height:0;overflow:auto;scroll-behavior:smooth}
    .messages{max-width:760px;margin:0 auto;padding:22px 20px 8px}
    .empty-state{max-width:600px;margin:9vh auto 0;padding:0 20px;text-align:center}
    .empty-state .mark{width:46px;height:46px;border-radius:13px;background:var(--ink);color:#fff;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:16px;margin:0 auto 18px}
    .empty-state h1{font-size:19px;font-weight:650;letter-spacing:-.01em;margin:0 0 8px}
    .empty-state p{font-size:13.5px;line-height:1.6;color:var(--ink-soft);margin:0 0 26px}
    .suggestions{display:grid;grid-template-columns:1fr 1fr;gap:9px}
    .suggestion{text-align:left;border:1px solid var(--line);background:var(--surface);border-radius:var(--radius-md);padding:12px 14px;transition:border-color .15s var(--ease),transform .15s var(--ease),box-shadow .15s var(--ease)}
    .suggestion:hover{border-color:#d5d3cd;box-shadow:var(--shadow-sm);transform:translateY(-1px)}
    .suggestion .t{display:block;font-size:12.5px;font-weight:600}
    .suggestion .d{display:block;font-size:11px;color:var(--muted);margin-top:3px;line-height:1.4}

    .message{display:flex;gap:10px;margin-bottom:18px;align-items:flex-start}
    .message.user{flex-direction:row-reverse}
    .message .avatar{width:26px;height:26px;border-radius:8px;background:var(--ink);color:#fff;display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;flex:none;margin-top:2px}
    .message.user .avatar{background:var(--accent-soft);color:var(--accent-strong)}
    .message-body{min-width:0;max-width:calc(100% - 70px)}
    .message.user .message-body{display:flex;flex-direction:column;align-items:flex-end}
    .bubble{display:inline-block;background:var(--surface);border:1px solid var(--line);border-radius:var(--radius-md);padding:11px 14px;font-size:13.5px;line-height:1.6;white-space:pre-wrap;text-align:left}
    .message.user .bubble{background:var(--ink);color:#fff;border-color:var(--ink)}
    .stamp{font-size:10.5px;color:var(--muted);margin-top:5px;padding:0 2px}
    .source-list{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
    .source-list a{font-size:10.5px;color:var(--accent-strong);text-decoration:none;border:1px solid var(--line);border-radius:var(--radius-pill);padding:4px 9px;background:var(--surface)}
    .source-list a:hover{border-color:var(--accent);background:var(--accent-soft)}

    .result-card{max-width:100%;margin:6px 0 20px;background:var(--surface);border:1px solid var(--line);border-radius:var(--radius-lg);padding:18px;box-shadow:var(--shadow-sm)}
    .result-head{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;margin-bottom:15px}
    .result-head h2{font-size:17px;font-weight:650;margin:0 0 3px;text-transform:capitalize;letter-spacing:-.01em}
    .result-head .stamp{margin:0;padding:0}
    .result-head strong{color:var(--accent-strong);font-size:13px;font-weight:700;white-space:nowrap;background:var(--accent-soft);padding:5px 10px;border-radius:var(--radius-pill)}
    .result-grid{display:grid;grid-template-columns:160px 1fr;gap:18px}
    .heatmap{width:160px;aspect-ratio:1;object-fit:cover;border-radius:var(--radius-md);background:var(--canvas);border:1px solid var(--line)}
    .prob{display:grid;grid-template-columns:1fr auto;gap:3px 8px;margin-bottom:7px;font-size:11px}
    .prob span{color:var(--ink-soft)}
    .prob b{font-weight:650}
    .track{grid-column:1/-1;height:5px;border-radius:var(--radius-pill);background:var(--line-soft);overflow:hidden}
    .track i{display:block;height:100%;background:var(--accent);border-radius:var(--radius-pill)}
    .notice{font-size:11.5px;color:var(--muted);margin:13px 0 0}
    .class-fact{font-size:12px;line-height:1.6;color:var(--ink-soft);margin:9px 0 0;padding-top:11px;border-top:1px solid var(--line-soft)}
    .class-fact a{margin-left:5px;font-size:10.5px;color:var(--accent-strong);text-decoration:none;border:1px solid var(--line);border-radius:var(--radius-pill);padding:3px 8px;white-space:nowrap}
    .class-fact a:hover{border-color:var(--accent);background:var(--accent-soft)}

    /* ---- composer ---- */
    .composer-zone{padding:12px 20px 16px;background:linear-gradient(var(--canvas) 40%,var(--canvas))}
    .composer-wrap{max-width:760px;margin:0 auto}
    .file-chip{display:none;align-items:center;gap:8px;margin:0 0 8px;background:var(--surface);border:1px solid var(--line);border-radius:var(--radius-md);padding:6px 8px 6px 6px;font-size:11.5px}
    .file-chip.show{display:flex}
    .file-chip img{width:26px;height:26px;border-radius:6px;object-fit:cover}
    .file-chip span{flex:1}
    .composer{border:1px solid var(--line);border-radius:22px;padding:6px;display:flex;align-items:flex-end;gap:6px;background:var(--surface);box-shadow:var(--shadow-sm);transition:border-color .15s var(--ease),box-shadow .15s var(--ease)}
    .composer:focus-within{border-color:#c9c6bf;box-shadow:var(--shadow-md)}
    .composer .attach{flex:none}
    .composer .attach input{display:none}
    .composer textarea{flex:1;border:0;resize:none;outline:0;background:transparent;min-height:22px;max-height:130px;padding:8px 4px;font-size:13.5px;line-height:1.5}
    .composer textarea::placeholder{color:var(--muted)}
    .send{flex:none;background:var(--ink);color:#fff;border-radius:50%}
    .send:disabled{background:var(--line);color:var(--muted)}
    .hint{text-align:center;color:var(--muted);font-size:10.5px;margin-top:9px}

    /* ---- process overlay ---- */
    .overlay{position:fixed;inset:0;z-index:50;background:rgba(20,20,22,.38);backdrop-filter:blur(6px);opacity:0;pointer-events:none;transition:opacity .22s var(--ease)}
    .overlay.open{opacity:1;pointer-events:auto}
    .drawer{position:absolute;right:18px;top:18px;bottom:18px;width:min(820px,calc(100% - 36px));background:var(--canvas);border:1px solid var(--line);border-radius:var(--radius-lg);box-shadow:var(--shadow-lg);transform:translateX(24px) scale(.99);transition:transform .25s var(--ease);display:grid;grid-template-rows:auto minmax(0,1fr);overflow:hidden}
    .overlay.open .drawer{transform:none}
    .drawer-head{display:flex;align-items:center;justify-content:space-between;padding:16px 20px;background:var(--surface);border-bottom:1px solid var(--line)}
    .drawer-title h2{margin:0;font-size:15px;font-weight:650}
    .drawer-title p{margin:2px 0 0;color:var(--muted);font-size:11.5px}
    .drawer-body{overflow:auto;padding:16px;display:grid;grid-template-columns:minmax(0,1.35fr) minmax(230px,.65fr);gap:12px;align-items:start}
    .card{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius-md);padding:15px}
    .card.full{grid-column:1/-1}
    .card-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:13px}
    .card-head strong{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--ink-soft)}
    .card-head span{font-size:10.5px;color:var(--muted);font-weight:560}

    /* graph */
    .graph-canvas{position:relative;height:230px}
    .graph-edges{position:absolute;inset:0;width:100%;height:100%;overflow:visible}
    .graph-edges path{fill:none;stroke:var(--line);stroke-width:1.6;transition:stroke .3s var(--ease),stroke-width .3s var(--ease)}
    .graph-edges path.active{stroke:var(--accent);stroke-width:2.2}
    .graph-edges path.done{stroke:#b9c3b3}
    .graph-edges path.seq{stroke-dasharray:4 3}
    .graph-edges path.seq.active{stroke-dasharray:none}
    .gnode{position:absolute;transform:translate(-50%,-50%);width:118px;background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:8px 10px;text-align:left;box-shadow:var(--shadow-sm);transition:border-color .25s var(--ease),box-shadow .25s var(--ease),background .25s var(--ease)}
    .gnode.hub{width:168px;text-align:center;background:var(--ink);border-color:var(--ink)}
    .gnode.hub strong,.gnode.hub span{color:#fff}
    .gnode.hub span{color:rgba(255,255,255,.62)}
    .gnode strong{display:block;font-size:11px;font-weight:700}
    .gnode span{display:block;font-size:9.5px;color:var(--muted);margin-top:2px;line-height:1.3}
    .gnode em{display:inline-flex;align-items:center;gap:4px;font-style:normal;font-size:8.5px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);margin-top:6px}
    .gnode em:before{content:"";width:5px;height:5px;border-radius:50%;background:var(--muted)}
    .gnode.active{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-soft)}
    .gnode.active em{color:var(--accent-strong)}
    .gnode.active em:before{background:var(--accent);animation:blink 1s ease-in-out infinite}
    .gnode.done{background:var(--good-soft);border-color:#bfe0cd}
    .gnode.done em{color:var(--good)}
    .gnode.done em:before{background:var(--good)}
    .gnode.error{background:var(--danger-soft);border-color:#e6b7c0}
    .gnode.error em{color:var(--danger)}
    .gnode.error em:before{background:var(--danger)}
    .gnode.hub.active{box-shadow:0 0 0 3px rgba(255,255,255,.25)}
    .gnode.hub.done{background:var(--ink);border-color:var(--good)}
    .gnode.hub.done em{color:#8fe0b8}
    .gnode.hub.done em:before{background:#8fe0b8}
    .gnode.hub.error{background:var(--ink);border-color:var(--danger)}
    .gnode.hub.error em{color:#f3a9b6}
    .gnode.hub.error em:before{background:#f3a9b6}

    .graph-list{display:none}
    .glist-row{display:flex;gap:11px;padding:9px 2px;position:relative}
    .glist-row:not(:last-child):before{content:"";position:absolute;left:13px;top:32px;bottom:-9px;border-left:1px dashed var(--line)}
    .glist-dot{width:26px;height:26px;border-radius:8px;background:var(--surface);border:1px solid var(--line);display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:700;flex:none;z-index:1}
    .glist-row.active .glist-dot{border-color:var(--accent);color:var(--accent-strong);box-shadow:0 0 0 3px var(--accent-soft)}
    .glist-row.done .glist-dot{background:var(--good-soft);border-color:#bfe0cd;color:var(--good)}
    .glist-row.error .glist-dot{background:var(--danger-soft);border-color:#e6b7c0;color:var(--danger)}
    .glist-row .gt{font-size:12px;font-weight:650}
    .glist-row .gd{font-size:10.5px;color:var(--muted);margin-top:1px}
    .glist-row .ge{font-size:9.5px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin-top:3px}

    .progress{height:6px;background:var(--line-soft);border-radius:var(--radius-pill);margin:14px 0 8px;overflow:hidden}
    .progress i{display:block;width:0;height:100%;background:var(--accent);border-radius:var(--radius-pill);transition:width .3s var(--ease)}
    .stage{font-size:11.5px;line-height:1.5;color:var(--ink-soft);min-height:32px}
    .trace{display:grid;max-height:420px;overflow:auto}
    .event{display:grid;grid-template-columns:22px 1fr;gap:9px;position:relative;padding-bottom:14px}
    .event:not(:last-child):before{content:"";position:absolute;left:10px;top:21px;bottom:0;border-left:1px solid var(--line)}
    .event-mark{width:20px;height:20px;border-radius:7px;background:var(--accent-soft);color:var(--accent-strong);display:flex;align-items:center;justify-content:center;font-size:8px;font-weight:700;z-index:1}
    .event p{margin:0;font-size:11.5px;line-height:1.5;color:var(--ink-soft)}
    .event strong{display:block;color:var(--ink);font-size:10.5px;margin-bottom:2px}
    .event time{font-size:9.5px;color:var(--muted)}
    .memory-grid{display:grid;gap:8px}
    .memory-item{background:var(--canvas);padding:10px 11px;border-radius:var(--radius-sm)}
    .memory-item strong{display:block;font-size:10.5px}
    .memory-item span{font-size:10px;line-height:1.45;color:var(--muted)}
    .empty{font-size:11.5px;color:var(--muted);padding:6px 0}

    /* ---- modals ---- */
    .modal-layer{position:fixed;inset:0;z-index:70;background:rgba(20,20,22,.42);backdrop-filter:blur(6px);display:flex;align-items:center;justify-content:center;padding:18px;opacity:0;pointer-events:none;transition:opacity .18s var(--ease)}
    .modal-layer.open{opacity:1;pointer-events:auto}
    .modal{width:min(480px,100%);max-height:calc(100vh - 36px);overflow:auto;background:var(--surface);border-radius:var(--radius-lg);box-shadow:var(--shadow-lg);transform:scale(.98);transition:transform .18s var(--ease)}
    .modal-layer.open .modal{transform:none}
    .modal-head{display:flex;justify-content:space-between;align-items:center;padding:16px 18px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--surface);z-index:1}
    .modal-head h2{margin:0;font-size:15px;font-weight:650}
    .modal-body{padding:18px}
    .field{margin-bottom:14px}
    .field label{display:block;font-size:11px;font-weight:650;margin-bottom:6px;color:var(--ink-soft)}
    .field input,.field select{width:100%;border:1px solid var(--line);border-radius:var(--radius-sm);padding:10px 11px;outline:0;background:var(--surface);font-size:13px;transition:border-color .15s var(--ease),box-shadow .15s var(--ease)}
    .field input:focus,.field select:focus{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-soft)}
    .field small{display:block;color:var(--muted);font-size:10.5px;line-height:1.5;margin-top:6px}
    .modal-actions{display:flex;justify-content:flex-end;gap:8px;margin-top:16px}
    .primary{border:0;background:var(--ink);color:#fff;border-radius:var(--radius-pill);padding:10px 18px;font-weight:600;font-size:13px;transition:background .15s var(--ease)}
    .primary:hover{background:var(--accent-strong)}
    .primary:disabled{background:var(--line);color:var(--muted)}
    .ghost-btn{border:1px solid var(--line);background:var(--surface);color:var(--ink);border-radius:var(--radius-pill);padding:10px 16px;font-weight:600;font-size:13px}
    .ghost-btn:hover{border-color:#d5d3cd;background:var(--canvas-alt)}
    .form-status{min-height:16px;font-size:11px;color:var(--muted);margin-top:10px}
    .form-status.error{color:var(--danger)}

    @media(max-width:900px){
      .drawer-body{grid-template-columns:1fr}
      .card.full{grid-column:auto}
      .drawer{right:10px;top:10px;bottom:10px;width:calc(100% - 20px)}
    }
    @media(max-width:680px){
      body{overflow:hidden}
      .shell{height:var(--app-height,100dvh)}
      .topbar{padding:10px 12px}
      .brand{margin-right:auto}
      .brand strong{display:none}
      .topbar-actions{gap:5px}
      .status-btn{padding:0 10px 0 9px;font-size:12px}
      .text-btn span{display:none}
      .text-btn{width:34px;padding:0;justify-content:center;border-radius:var(--radius-pill)}
      .popover{right:12px;left:12px;width:auto}
      .messages{padding:16px 14px 4px}
      .empty-state{margin-top:6vh;padding:0 10px}
      .suggestions{grid-template-columns:1fr}
      .message-body{max-width:calc(100% - 56px)}
      .composer-zone{padding:10px 12px 14px}
      .result-card{padding:14px}
      .result-grid{grid-template-columns:1fr}
      .heatmap{width:100%;max-width:220px;margin:0 auto}
      .drawer{inset:auto 0 0;width:100%;height:92dvh;max-height:100dvh;border-radius:20px 20px 0 0;transform:translateY(30px)}
      .drawer-body{padding:12px}
      .graph-canvas{display:none}
      .graph-list{display:block}
    }
  </style>
</head>
<body>
<div class="shell">
  <header class="topbar">
    <div class="brand"><div class="mark">AD</div><strong>AgenticDerma</strong></div>
    <div class="topbar-actions">
      <button class="status-btn" id="statusButton" aria-haspopup="true" aria-expanded="false">
        <span class="status-dot" id="statusDot"></span><span id="statusLabel">System</span>
        <svg class="icon chev" viewBox="0 0 20 20"><path d="M5 8l5 5 5-5"/></svg>
      </button>
      <button class="text-btn" id="connectionButton" title="Model connection">
        <svg class="icon" viewBox="0 0 20 20" style="width:15px;height:15px"><path d="M10 2v4M10 14v4M4.9 4.9l2.8 2.8M12.3 12.3l2.8 2.8M2 10h4M14 10h4M4.9 15.1l2.8-2.8M12.3 7.7l2.8-2.8"/></svg>
        <span id="providerLabel">Model</span>
      </button>
      <button class="text-btn process-button" id="processButton" title="View process">
        <svg class="icon" viewBox="0 0 20 20" style="width:15px;height:15px"><circle cx="5" cy="10" r="2"/><circle cx="15" cy="5" r="2"/><circle cx="15" cy="15" r="2"/><path d="M6.7 9.2l6.6-3.4M6.7 10.8l6.6 3.4"/></svg>
        <span>Process</span>
      </button>
      <button class="icon-btn subtle" id="newChat" title="New chat" aria-label="New chat">
        <svg class="icon" viewBox="0 0 20 20"><path d="M10 4v12M4 10h12"/></svg>
      </button>
    </div>
    <div class="popover" id="statusPopover" role="menu">
      <div class="popover-row"><span class="status-dot" id="dotModel"></span><span class="label">Classifier</span><span class="value" id="valModel">…</span></div>
      <div class="popover-row"><span class="status-dot" id="dotData"></span><span class="label">Dataset</span><span class="value" id="valData">…</span></div>
      <div class="popover-row"><span class="status-dot" id="dotKnowledge"></span><span class="label">Knowledge</span><span class="value" id="valKnowledge">…</span></div>
      <div class="popover-divider"></div>
      <div class="popover-row"><span class="status-dot" id="dotLLM"></span><span class="label">Model connection</span><span class="value" id="valLLM">…</span></div>
    </div>
  </header>

  <div class="stage-area" id="stageArea">
    <div class="empty-state" id="emptyState">
      <div class="mark">AD</div>
      <h1>Ask AgenticDerma</h1>
      <p>Send a dermoscopic image for a model result, or ask about the classifier, its training, or dermoscopy concepts. Every claim is grounded in a recorded evaluation or a cited passage before it reaches you.</p>
      <div class="suggestions">
        <button class="suggestion" data-prompt="Is the trained model ready?"><span class="t">Check model</span><span class="d">Readiness and test accuracy</span></button>
        <button class="suggestion" data-prompt="Train the models automatically."><span class="t">Start training</span><span class="d">Full process, all three candidates</span></button>
        <button class="suggestion" data-prompt="Explain the latest result."><span class="t">Explain result</span><span class="d">Probabilities and attribution</span></button>
        <button class="suggestion" data-prompt="What can I ask you?"><span class="t">What can I ask?</span><span class="d">Capabilities overview</span></button>
      </div>
    </div>
    <section class="messages" id="messages"></section>
  </div>

  <footer class="composer-zone">
    <div class="composer-wrap">
      <div class="file-chip" id="fileChip"><img id="filePreview" alt="Selected image"><span id="fileName"></span>
        <button class="icon-btn subtle" id="removeFile" aria-label="Remove image" style="width:24px;height:24px">
          <svg class="icon" viewBox="0 0 20 20" style="width:13px;height:13px"><path d="M5 5l10 10M15 5L5 15"/></svg>
        </button>
      </div>
      <div class="composer">
        <label class="icon-btn attach" title="Attach image" aria-label="Attach image">
          <svg class="icon" viewBox="0 0 20 20"><path d="M10 5v10M5 10h10"/></svg>
          <input id="file" type="file" accept="image/png,image/jpeg,image/webp,image/bmp">
        </label>
        <textarea id="prompt" rows="1" placeholder="Ask AgenticDerma or attach an image…"></textarea>
        <button class="icon-btn send" id="send" aria-label="Send">
          <svg class="icon" viewBox="0 0 20 20" style="stroke:#fff"><path d="M10 15V5M5 9l5-5 5 5"/></svg>
        </button>
      </div>
      <div class="hint">Enter to send · Shift + Enter for a new line</div>
    </div>
  </footer>
</div>

<div class="overlay" id="processOverlay" aria-hidden="true">
  <section class="drawer" role="dialog" aria-modal="true" aria-labelledby="processTitle">
    <header class="drawer-head">
      <div class="drawer-title"><h2 id="processTitle">System activity</h2><p>Current route, agent work, and handoffs</p></div>
      <button class="icon-btn" id="closeProcess" aria-label="Close">
        <svg class="icon" viewBox="0 0 20 20"><path d="M5 5l10 10M15 5L5 15"/></svg>
      </button>
    </header>
    <div class="drawer-body">
      <section class="card full">
        <div class="card-head"><strong>Process map</strong><span id="graphState">Idle</span></div>

        <div class="graph-canvas">
          <svg class="graph-edges" viewBox="0 0 100 100" preserveAspectRatio="none">
            <path class="hub-edge" data-edge="Agent1" d="M50,23 L10,71"/>
            <path class="hub-edge" data-edge="Agent2" d="M50,23 L36.6,71"/>
            <path class="hub-edge" data-edge="Agent3" d="M50,23 L63.3,71"/>
            <path class="hub-edge" data-edge="Agent4" d="M50,23 L90,71"/>
            <path class="seq" data-seq="Agent1" d="M15.5,85 L31.1,85"/>
            <path class="seq" data-seq="Agent2" d="M42.1,85 L57.8,85"/>
            <path class="seq" data-seq="Agent3" d="M68.8,85 L84.5,85"/>
          </svg>
          <div class="gnode hub" data-agent="AgenticDerma" style="left:50%;top:16%"><strong>AgenticDerma</strong><span>Route, memory, response</span><em id="emHub">Idle</em></div>
          <div class="gnode" data-agent="Agent1" style="left:10%;top:78%"><strong>Agent1</strong><span>Hyperparameter planning</span><em id="emAgent1">Idle</em></div>
          <div class="gnode" data-agent="Agent2" style="left:36.6%;top:78%"><strong>Agent2</strong><span>Adaptive training</span><em id="emAgent2">Idle</em></div>
          <div class="gnode" data-agent="Agent3" style="left:63.3%;top:78%"><strong>Agent3</strong><span>Evaluation, prediction</span><em id="emAgent3">Idle</em></div>
          <div class="gnode" data-agent="Agent4" style="left:90%;top:78%"><strong>Agent4</strong><span>Attribution, evidence</span><em id="emAgent4">Idle</em></div>
        </div>

        <div class="graph-list" id="graphList">
          <div class="glist-row" data-agent="AgenticDerma"><div class="glist-dot">AD</div><div><div class="gt">AgenticDerma</div><div class="gd">Route, memory, response</div><div class="ge">Idle</div></div></div>
          <div class="glist-row" data-agent="Agent1"><div class="glist-dot">1</div><div><div class="gt">Agent1</div><div class="gd">Hyperparameter planning</div><div class="ge">Idle</div></div></div>
          <div class="glist-row" data-agent="Agent2"><div class="glist-dot">2</div><div><div class="gt">Agent2</div><div class="gd">Adaptive training</div><div class="ge">Idle</div></div></div>
          <div class="glist-row" data-agent="Agent3"><div class="glist-dot">3</div><div><div class="gt">Agent3</div><div class="gd">Evaluation, prediction</div><div class="ge">Idle</div></div></div>
          <div class="glist-row" data-agent="Agent4"><div class="glist-dot">4</div><div><div class="gt">Agent4</div><div class="gd">Attribution, evidence</div><div class="ge">Idle</div></div></div>
        </div>

        <div class="progress"><i id="progress"></i></div><div class="stage" id="stage">No process is running.</div>
      </section>
      <section class="card"><div class="card-head"><strong>Communication</strong><span>Live trace</span></div><div class="trace" id="activity"><div class="empty">Agent communication will appear when a process starts.</div></div></section>
      <section class="card"><div class="card-head"><strong>Memory</strong><span>Two layers</span></div><div class="memory-grid"><div class="memory-item"><strong>Deterministic state</strong><span>Model readiness, metrics, agent state, and the latest result.</span></div><div class="memory-item"><strong>Working context</strong><span>Relevant events are sampled by query, session, and recency, with the selection seed recorded.</span></div><div class="memory-item"><strong>Knowledge retrieval</strong><span id="knowledgeDetail">Loading the dermoscopy index.</span></div></div></section>
    </div>
  </section>
</div>

<div class="modal-layer" id="connectionModal" aria-hidden="true">
  <section class="modal" role="dialog" aria-modal="true" aria-labelledby="connectionTitle">
    <header class="modal-head"><h2 id="connectionTitle">Model connection</h2>
      <button class="icon-btn" id="closeConnection" aria-label="Close"><svg class="icon" viewBox="0 0 20 20"><path d="M5 5l10 10M15 5L5 15"/></svg></button>
    </header>
    <div class="modal-body">
      <div class="field"><label for="provider">Provider</label>
        <select id="provider">
          <option value="ollama">Local Ollama (no account or key)</option>
          <option value="groq">Groq (account and key required)</option>
          <option value="gemini">Google Gemini (account and key required)</option>
          <option value="openrouter">OpenRouter (account and key required)</option>
          <option value="compatible">OpenAI-compatible API</option>
        </select>
      </div>
      <div class="field"><label for="endpoint">Chat endpoint</label><input id="endpoint" autocomplete="off"><small>Online endpoints must use HTTPS. Local HTTP is accepted for Ollama, LM Studio, and similar servers.</small></div>
      <div class="field"><label for="modelName">Model</label><input id="modelName" autocomplete="off"></div>
      <div class="field" id="keyField"><label for="apiKey">API key</label><input id="apiKey" type="password" autocomplete="off" placeholder="Paste the key for this session"><small>The key is held in server memory until the platform stops. It is not written to the project or returned by the API.</small></div>
      <div class="form-status" id="connectionStatus"></div>
      <div class="modal-actions"><button class="ghost-btn" id="cancelConnection">Cancel</button><button class="primary" id="saveConnection">Test and use</button></div>
    </div>
  </section>
</div>

<div class="modal-layer" id="trainingModal" aria-hidden="true">
  <section class="modal" role="dialog" aria-modal="true" aria-labelledby="trainingTitle">
    <header class="modal-head"><h2 id="trainingTitle">Training plan</h2>
      <button class="icon-btn" id="closeTraining" aria-label="Close"><svg class="icon" viewBox="0 0 20 20"><path d="M5 5l10 10M15 5L5 15"/></svg></button>
    </header>
    <div class="modal-body">
      <div class="field"><label for="targetAccuracy">Target validation accuracy</label><input id="targetAccuracy" type="number" min="0.01" max="1" step="0.01" value="0.70"><small>Training may stop after reaching this value. The epoch limit always applies.</small></div>
      <div class="field"><label for="maxEpochs">Maximum epochs per candidate</label><input id="maxEpochs" type="number" min="1" max="100" step="1" value="3"></div>
      <div class="field"><label for="tuningMode">Hyperparameters</label><select id="tuningMode"><option value="auto">Automatic (Agent1 selects and tunes them)</option><option value="manual">Manual (I set them)</option></select><small>Automatic lets Agent1 pick each starting rate from prior runs and Agent2 adjust it after every epoch. Manual keeps your values fixed for the whole run.</small></div>
      <div id="manualFields" style="display:none">
        <div class="field"><label for="learningRate">Learning rate</label><input id="learningRate" type="number" min="0.00005" max="0.004" step="0.0001" placeholder="0.003"></div>
        <div class="field"><label for="weightDecay">Weight decay</label><input id="weightDecay" type="number" min="0" max="0.1" step="0.0001" placeholder="0.0001"></div>
        <div class="field"><label for="batchSize">Batch size</label><input id="batchSize" type="number" min="8" max="512" step="1" placeholder="128"></div>
      </div>
      <div class="form-status" id="trainingStatus"></div>
      <div class="modal-actions"><button class="ghost-btn" id="cancelTraining">Cancel</button><button class="primary" id="startTraining">Start full process</button></div>
    </div>
  </section>
</div>

<script>
const $=s=>document.querySelector(s),messages=$('#messages'),promptBox=$('#prompt'),sendButton=$('#send'),emptyState=$('#emptyState');
let session=crypto.randomUUID(),selected=null,busy=false,currentJob=null,currentLLM=null,pendingTraining=null,lastHealth=null;
const presets={
  ollama:{endpoint:'http://127.0.0.1:11434/api/chat',model:'qwen3:1.7b'},
  groq:{endpoint:'https://api.groq.com/openai/v1/chat/completions',model:'openai/gpt-oss-20b'},
  gemini:{endpoint:'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions',model:'gemini-2.5-flash'},
  openrouter:{endpoint:'https://openrouter.ai/api/v1/chat/completions',model:'openrouter/free'},
  compatible:{endpoint:'https://api.openai.com/v1/chat/completions',model:''},
};
function clock(){return new Date().toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'})}
function sizeApp(){const height=Math.round(window.visualViewport?.height||window.innerHeight);document.documentElement.style.setProperty('--app-height',height+'px')}
function toggleEmpty(){emptyState.style.display=messages.children.length?'none':'block'}
function addMessage(role,text,sources=[]){
  const row=document.createElement('div');row.className='message '+(role==='user'?'user':'');
  row.innerHTML='<div class="avatar"></div><div class="message-body"><div class="bubble"></div><div class="stamp"></div><div class="source-list"></div></div>';
  row.querySelector('.avatar').textContent=role==='user'?'You':'AD';
  row.querySelector('.bubble').textContent=text;
  row.querySelector('.stamp').textContent=(role==='user'?'You':'AgenticDerma')+' · '+clock();
  const sourceBox=row.querySelector('.source-list');
  sources.forEach(source=>{const link=document.createElement('a');link.href=source.url;link.target='_blank';link.rel='noopener';link.textContent=source.label+' · '+source.title;link.title=source.citation||source.title;sourceBox.appendChild(link)});
  messages.appendChild(row);toggleEmpty();$('#stageArea').scrollTop=$('#stageArea').scrollHeight;
}
function setBusy(value){busy=value;sendButton.disabled=value;promptBox.disabled=value;$('#processButton').classList.toggle('running',value)}
function openLayer(element){element.classList.add('open');element.setAttribute('aria-hidden','false')}
function closeLayer(element){element.classList.remove('open');element.setAttribute('aria-hidden','true')}
function openProcess(){openLayer($('#processOverlay'))}
function closeProcess(){closeLayer($('#processOverlay'))}
function renderGraph(graph={}){
  document.querySelectorAll('[data-agent]').forEach(node=>{
    const state=graph[node.dataset.agent]||'idle';
    node.classList.toggle('active',state==='active');node.classList.toggle('done',state==='done');node.classList.toggle('error',state==='error');
    const label=node.querySelector('em')||node.querySelector('.ge');if(label)label.textContent=state;
  });
  document.querySelectorAll('.hub-edge').forEach(edge=>{
    const state=graph[edge.dataset.edge]||'idle';
    edge.classList.toggle('active',state==='active');edge.classList.toggle('done',state==='done'||state==='error');
  });
  const order=['Agent1','Agent2','Agent3','Agent4'];
  document.querySelectorAll('.seq').forEach(edge=>{
    const from=edge.dataset.seq,to=order[order.indexOf(from)+1];
    const fromState=graph[from]||'idle',toState=graph[to]||'idle';
    edge.classList.toggle('done',fromState==='done');
    edge.classList.toggle('active',fromState==='done'&&toState==='active'||toState==='done'&&fromState==='done');
  });
}
function renderActivity(items=[]){
  const box=$('#activity');box.innerHTML='';
  if(!items.length){box.innerHTML='<div class="empty">Agent communication will appear when a process starts.</div>';return}
  items.slice(-24).reverse().forEach(item=>{
    const event=document.createElement('div');event.className='event';
    event.innerHTML='<div class="event-mark"></div><div><strong></strong><p></p><time></time></div>';
    event.querySelector('.event-mark').textContent=item.actor==='AgenticDerma'?'AD':item.actor.replace('Agent','A');
    event.querySelector('strong').textContent=item.actor;event.querySelector('p').textContent=item.message;event.querySelector('time').textContent=item.time||'';
    box.appendChild(event);
  });
}
function renderResult(data,job){
  const card=document.createElement('section');card.className='result-card';
  const rows=Object.entries(data.probabilities).sort((a,b)=>b[1]-a[1]).map(([name,value])=>'<div class="prob"><span>'+name+'</span><b>'+(value*100).toFixed(1)+'%</b><div class="track"><i style="width:'+(value*100)+'%"></i></div></div>').join('');
  card.innerHTML='<div class="result-head"><div><h2></h2><span class="stamp">Final output</span></div><strong class="confidence"></strong></div><div class="result-grid"><img class="heatmap" alt="Input-gradient attribution"><div>'+rows+'</div></div><p class="notice"></p><p class="class-fact"></p>';
  card.querySelector('h2').textContent=data.prediction;
  card.querySelector('.confidence').textContent=(data.confidence*100).toFixed(1)+'% confidence';
  card.querySelector('.heatmap').src='/api/file?job='+encodeURIComponent(job)+'&name=attribution.png&t='+Date.now();
  card.querySelector('.notice').textContent=data.uncertain?'The uncertainty rule was triggered.':'The uncertainty rule was not triggered.';
  const factBox=card.querySelector('.class-fact');factBox.textContent=data.class_information+' ';
  (data.citations||[]).forEach(source=>{const link=document.createElement('a');link.href=source.url;link.target='_blank';link.rel='noopener';link.textContent=source.title;factBox.appendChild(link)});
  messages.appendChild(card);toggleEmpty();$('#stageArea').scrollTop=$('#stageArea').scrollHeight;
}
function setStatusPill(level,text){
  $('#statusDot').className='status-dot '+level;$('#statusLabel').textContent=text;
}
async function refreshHealth(silent){
  try{
    const health=await fetch('/api/health').then(r=>r.json());
    lastHealth=health;currentLLM=health.language_model;
    const rows=[
      ['dotModel','valModel',health.model,health.model?'Ready':'Not trained'],
      ['dotData','valData',health.data,health.data?'Ready':'Downloads when needed'],
      ['dotKnowledge','valKnowledge',health.knowledge.ready,health.knowledge.ready?health.knowledge.sources+' sources':'Not ready'],
      ['dotLLM','valLLM',health.llm,health.llm?health.language_model.provider_label:'Not connected'],
    ];
    rows.forEach(([dotId,valId,ok,text])=>{$('#'+dotId).className='status-dot '+(ok?'good':'warn');$('#'+valId).textContent=text});
    $('#providerLabel').textContent=health.language_model.provider_label;
    $('#knowledgeDetail').textContent=health.knowledge.passages+' indexed passages from '+health.knowledge.sources+' sources.';
    if(!health.model){setStatusPill('warn','Setup needed')}
    else if(busy){setStatusPill('busy','Working')}
    else{setStatusPill('good','Ready')}
  }catch(error){
    if(!silent)setStatusPill('warn','Unavailable');
    ['dotModel','dotData','dotKnowledge','dotLLM'].forEach(id=>$('#'+id).className='status-dot warn');
  }
}
async function poll(job){
  currentJob=job;
  const response=await fetch('/api/status?id='+encodeURIComponent(job));
  const state=await response.json();
  if(!response.ok)throw new Error(state.error||'Process status is unavailable');
  $('#progress').style.width=(state.percent||0)+'%';
  $('#stage').textContent=state.stage||'Working';
  $('#graphState').textContent=state.status==='complete'?'Complete':state.status==='error'?'Stopped':'Running';
  setStatusPill('busy','Working');
  renderGraph(state.graph);renderActivity(state.activities);
  if(state.status==='complete'){
    setBusy(false);addMessage('assistant',state.message,state.sources||[]);
    if(state.result)renderResult(state.result,job);
    await refreshHealth(true);currentJob=null;return;
  }
  if(state.status==='error'){
    setBusy(false);addMessage('assistant','The process stopped before producing a result. '+state.error);
    await refreshHealth(true);currentJob=null;return;
  }
  setTimeout(()=>poll(job).catch(fail),500);
}
function fail(error){setBusy(false);addMessage('assistant','I could not complete that request. '+(error.message||error))}
async function submit(){
  if(busy)return;
  const message=promptBox.value.trim();const attachment=selected;
  if(!message&&!attachment)return;
  addMessage('user',message||(attachment?'Analyze this image.':''));
  promptBox.value='';promptBox.style.height='';chooseFile(null);setBusy(true);
  const payload={session,message,name:attachment?.name,data:attachment?.data};
  try{
    const response=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const data=await response.json();
    if(!response.ok)throw new Error(data.error||'Request failed');
    if(data.message)addMessage('assistant',data.message,data.sources||[]);
    if(data.job){openProcess();poll(data.job).catch(fail)}
    else if(data.action==='train'){pendingTraining=payload;setBusy(false);openLayer($('#trainingModal'))}
    else setBusy(false);
  }catch(error){fail(error)}
}
function chooseFile(file){
  selected=null;$('#fileChip').classList.remove('show');$('#file').value='';
  if(!file)return;
  if(file.size>10*1024*1024){fail(new Error('Choose an image no larger than 10 MB.'));return}
  if(!['image/png','image/jpeg','image/webp','image/bmp'].includes(file.type)){fail(new Error('Use a PNG, JPG, WebP, or BMP image.'));return}
  const reader=new FileReader();
  reader.onload=()=>{selected={name:file.name,data:reader.result};$('#fileName').textContent=file.name;$('#filePreview').src=reader.result;$('#fileChip').classList.add('show')};
  reader.readAsDataURL(file);
}
function applyProviderPreset(force){
  const provider=$('#provider').value,preset=presets[provider];
  if(force||!$('#endpoint').value)$('#endpoint').value=preset.endpoint;
  if(force||!$('#modelName').value)$('#modelName').value=preset.model;
  $('#keyField').style.display=provider==='ollama'?'none':'block';
}
async function openConnection(){
  if(!currentLLM)await refreshHealth(true);
  $('#provider').value=currentLLM?.provider||'ollama';$('#endpoint').value=currentLLM?.endpoint||'';$('#modelName').value=currentLLM?.model||'';$('#apiKey').value='';
  $('#connectionStatus').textContent=currentLLM?.key_loaded?'A key is already loaded for this session.':'';
  $('#connectionStatus').className='form-status';
  applyProviderPreset(false);openLayer($('#connectionModal'));
}
async function saveConnection(){
  const button=$('#saveConnection'),status=$('#connectionStatus');
  button.disabled=true;status.className='form-status';status.textContent='Testing the connection…';
  try{
    const response=await fetch('/api/llm/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({provider:$('#provider').value,endpoint:$('#endpoint').value,model:$('#modelName').value,apiKey:$('#apiKey').value})});
    const data=await response.json();
    if(!response.ok)throw new Error(data.error||'Connection failed');
    currentLLM=data.configuration;status.textContent=data.message;
    await refreshHealth(true);setTimeout(()=>closeLayer($('#connectionModal')),700);
  }catch(error){status.textContent=error.message;status.className='form-status error'}
  finally{button.disabled=false}
}
function applyTuningMode(){$('#manualFields').style.display=$('#tuningMode').value==='manual'?'block':'none'}
async function startTraining(){
  const status=$('#trainingStatus'),button=$('#startTraining');
  const targetAccuracy=Number($('#targetAccuracy').value),maxEpochs=Number($('#maxEpochs').value),tuning=$('#tuningMode').value;
  if(!(targetAccuracy>0&&targetAccuracy<=1)){status.textContent='Enter a target greater than 0 and no more than 1.';status.className='form-status error';return}
  if(!Number.isInteger(maxEpochs)||maxEpochs<1||maxEpochs>100){status.textContent='Enter an epoch limit from 1 to 100.';status.className='form-status error';return}
  if(tuning==='manual'&&!$('#learningRate').value){status.textContent='Enter a learning rate for manual tuning.';status.className='form-status error';return}
  button.disabled=true;status.textContent='Starting the training process…';status.className='form-status';setBusy(true);
  try{
    const payload={...(pendingTraining||{session}),mode:'full',targetAccuracy,maxEpochs,tuning,learningRate:$('#learningRate').value,weightDecay:$('#weightDecay').value,batchSize:$('#batchSize').value};
    const response=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const data=await response.json();
    if(!response.ok)throw new Error(data.error||'Training could not start');
    pendingTraining=null;closeLayer($('#trainingModal'));openProcess();poll(data.job).catch(fail);
  }catch(error){setBusy(false);status.textContent=error.message;status.className='form-status error'}
  finally{button.disabled=false}
}
document.querySelectorAll('[data-prompt]').forEach(button=>button.onclick=()=>{promptBox.value=button.dataset.prompt;submit()});
sendButton.onclick=submit;
promptBox.onkeydown=event=>{if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();submit()}};
promptBox.oninput=()=>{promptBox.style.height='';promptBox.style.height=Math.min(130,promptBox.scrollHeight)+'px'};
$('#file').onchange=event=>chooseFile(event.target.files[0]);
$('#removeFile').onclick=()=>chooseFile(null);
$('#newChat').onclick=()=>{
  session=crypto.randomUUID();selected=null;pendingTraining=null;chooseFile(null);setBusy(false);
  messages.innerHTML='';toggleEmpty();
  $('#progress').style.width='0';$('#stage').textContent='No process is running.';$('#graphState').textContent='Idle';
  renderActivity([]);renderGraph({});
};
$('#processButton').onclick=openProcess;
$('#closeProcess').onclick=closeProcess;
$('#processOverlay').onclick=event=>{if(event.target===event.currentTarget)closeProcess()};
$('#statusButton').onclick=event=>{
  event.stopPropagation();
  const open=$('#statusPopover').classList.toggle('open');
  $('#statusButton').setAttribute('aria-expanded',open?'true':'false');
  if(open)refreshHealth(true);
};
document.addEventListener('click',event=>{
  if(!$('#statusPopover').contains(event.target)&&event.target!==$('#statusButton')&&!$('#statusButton').contains(event.target)){
    $('#statusPopover').classList.remove('open');$('#statusButton').setAttribute('aria-expanded','false');
  }
});
$('#connectionButton').onclick=openConnection;
$('#closeConnection').onclick=()=>closeLayer($('#connectionModal'));
$('#cancelConnection').onclick=()=>closeLayer($('#connectionModal'));
$('#connectionModal').onclick=event=>{if(event.target===event.currentTarget)closeLayer(event.currentTarget)};
$('#provider').onchange=()=>applyProviderPreset(true);
$('#saveConnection').onclick=saveConnection;
$('#tuningMode').onchange=applyTuningMode;
$('#startTraining').onclick=startTraining;
$('#closeTraining').onclick=()=>closeLayer($('#trainingModal'));
$('#cancelTraining').onclick=()=>{pendingTraining=null;closeLayer($('#trainingModal'))};
$('#trainingModal').onclick=event=>{if(event.target===event.currentTarget)closeLayer(event.currentTarget)};
document.addEventListener('keydown',event=>{
  if(event.key==='Escape'){
    closeProcess();closeLayer($('#connectionModal'));closeLayer($('#trainingModal'));
    $('#statusPopover').classList.remove('open');$('#statusButton').setAttribute('aria-expanded','false');
  }
});
toggleEmpty();
sizeApp();setTimeout(sizeApp,100);setTimeout(sizeApp,500);
window.addEventListener('resize',sizeApp);window.visualViewport?.addEventListener('resize',sizeApp);
refreshHealth();setInterval(()=>refreshHealth(true),15000);
</script>
</body></html>"""


def graph_state() -> dict[str, str]:
    return {name: "idle" for name in ("AgenticDerma", "Agent1", "Agent2", "Agent3", "Agent4")}


def update_job(job_id: str, **values: Any) -> None:
    with JOBS_LOCK:
        JOBS[job_id].update(values)


def clean_activity(actor: str, message: str) -> str:
    text = re.sub(rf"^{re.escape(actor)}(?:'s|\s+is|\s*[·:])?\s*", "", message, flags=re.IGNORECASE).strip()
    if not text:
        return "Working."
    return text[0].upper() + text[1:]


def activity(job_id: str, actor: str, message: str) -> None:
    with JOBS_LOCK:
        items = list(JOBS[job_id].get("activities", []))
        items.append({"actor": actor, "message": clean_activity(actor, message), "time": project.now_iso()[11:19]})
        JOBS[job_id]["activities"] = items[-50:]


def actor_for(stage: str) -> str:
    for name in ("Agent1", "Agent2", "Agent3", "Agent4"):
        if name.lower() in stage.lower():
            return name
    return "AgenticDerma"


def uploaded_image(payload: dict[str, Any], output_dir: Path) -> Path | None:
    encoded = payload.get("data")
    if not encoded:
        return None
    encoded = str(encoded)
    if "," not in encoded:
        raise ValueError("Invalid image upload")
    raw = base64.b64decode(encoded.split(",", 1)[1], validate=True)
    if not raw or len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("Image must be no more than 10 MB")
    suffix = Path(str(payload.get("name", "image.png"))).suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        raise ValueError("Use a PNG, JPG, WebP, or BMP image")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / ("input" + suffix)
    path.write_bytes(raw)
    project.validate_input_image(path)
    return path


def run_job(
    job_id: str, image_path: Path, output_dir: Path, mode: str,
    target_accuracy: float, max_epochs: int, session: str,
    hyperparameters: dict[str, Any] | None = None,
) -> None:
    try:
        previous_actor = "AgenticDerma"
        previous_stage = "Accepted the request and selected the execution path."

        def progress(percent: int, stage: str) -> None:
            nonlocal previous_actor, previous_stage
            actor = actor_for(stage)
            with JOBS_LOCK:
                graph = dict(JOBS[job_id]["graph"])
            if previous_actor != actor:
                graph[previous_actor] = "done"
                handoff = COORDINATOR.handoff(session, previous_actor, actor, previous_stage)
                activity(job_id, previous_actor, handoff["content"])
            graph[actor] = "active"
            update_job(job_id, percent=percent, stage=stage, graph=graph)
            previous_actor = actor
            previous_stage = stage
            activity(job_id, actor, stage)

        activity(job_id, "AgenticDerma", "Accepted the request and selected the execution path.")
        with MODEL_LOCK:
            if mode == "full":
                completed = project.run_full_process(image_path, target_accuracy, max_epochs, output_dir, progress, hyperparameters)
                result, summary = completed["prediction"], completed["summary"]
            else:
                config = project.load_config()
                manifest_path = project.deliverable_path("D4.3")
                if not manifest_path.is_file():
                    raise FileNotFoundError("No trained model was found. Start a training run first.")
                result = project.run_user_image(
                    image_path, project.read_json(manifest_path), config,
                    torch.device("cuda" if torch.cuda.is_available() else "cpu"), output_dir, progress,
                )
                summary = None
        message = COORDINATOR.finish_prediction(session, result, summary)
        with JOBS_LOCK:
            graph = dict(JOBS[job_id]["graph"])
        graph = {name: "done" if state in {"active", "done"} else state for name, state in graph.items()}
        graph["AgenticDerma"] = "done"
        activity(job_id, "AgenticDerma", "Verified the result and returned it to the session.")
        update_job(
            job_id, status="complete", percent=100, stage="Result ready", result=result,
            summary=summary, message=message, sources=[], graph=graph,
        )
    except Exception as exc:
        with JOBS_LOCK:
            graph = dict(JOBS[job_id].get("graph", graph_state()))
            current_stage = str(JOBS[job_id].get("stage", ""))
        graph[actor_for(current_stage)] = "error"
        activity(job_id, "AgenticDerma", f"The process stopped because {exc}")
        update_job(job_id, status="error", stage="Process stopped", error=str(exc), graph=graph)


def manual_hyperparameters(payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any] | None:
    if payload.get("tuning") != "manual":
        return None
    if payload.get("learningRate") in (None, ""):
        raise ValueError("Enter a learning rate for manual tuning")
    values = {"learning_rate": float(payload["learningRate"])}
    if payload.get("weightDecay") not in (None, ""):
        values["weight_decay"] = float(payload["weightDecay"])
    if payload.get("batchSize") not in (None, ""):
        values["batch_size"] = int(payload["batchSize"])
    project.validate_manual_hyperparameters(values, config)
    return values


def start_job(payload: dict[str, Any], mode: str, session: str) -> str:
    config = project.load_config()
    target_accuracy = float(payload.get("targetAccuracy", config["training"]["default_target_accuracy"]))
    max_epochs = int(payload.get("maxEpochs", config["training"]["default_max_epochs"]))
    if not 0 < target_accuracy <= 1:
        raise ValueError("Target accuracy must be greater than 0 and no more than 1")
    if not 1 <= max_epochs <= 100:
        raise ValueError("Maximum epochs must be from 1 to 100")
    hyperparameters = manual_hyperparameters(payload, config) if mode == "full" else None
    job_id = uuid.uuid4().hex
    output_dir = OUTPUT_ROOT / "sessions" / session / job_id
    image_path = uploaded_image(payload, output_dir)
    if image_path is None:
        if mode == "full" and DEFAULT_IMAGE.is_file():
            image_path = DEFAULT_IMAGE
        else:
            raise ValueError("Attach an image to run Final Output")
    job: dict[str, Any] = {
        "status": "running", "percent": 1, "stage": "Preparing the request", "mode": mode,
        "session": session, "output_dir": str(output_dir), "graph": graph_state(), "activities": [],
    }
    job["graph"]["AgenticDerma"] = "active"
    with JOBS_LOCK:
        JOBS[job_id] = job
    threading.Thread(
        target=run_job,
        args=(job_id, image_path, output_dir, mode, target_accuracy, max_epochs, session, hyperparameters),
        daemon=True,
    ).start()
    return job_id


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def send_bytes(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status: int, payload: object) -> None:
        self.send_bytes(status, "application/json; charset=utf-8", json.dumps(payload, ensure_ascii=False).encode())

    def read_payload(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_UPLOAD_BYTES * 2:
            raise ValueError("Request is empty or too large")
        return json.loads(self.rfile.read(length))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_bytes(200, "text/html; charset=utf-8", PAGE.encode())
            return
        if parsed.path == "/favicon.ico":
            self.send_bytes(204, "image/x-icon", b"")
            return
        if parsed.path in {"/api/health", "/api/status"} and not parsed.query:
            self.send_json(200, COORDINATOR.status())
            return
        if parsed.path == "/api/llm":
            self.send_json(200, COORDINATOR.llm.status())
            return
        query = parse_qs(parsed.query)
        if parsed.path == "/api/status":
            job_id = query.get("id", [""])[0]
            with JOBS_LOCK:
                job = dict(JOBS.get(job_id, {}))
            self.send_json(200 if job else 404, job or {"error": "Job not found"})
            return
        if parsed.path == "/api/memory":
            session = query.get("session", ["system"])[0]
            self.send_json(200, COORDINATOR.memory.context(session, "current session"))
            return
        if parsed.path == "/api/file":
            job_id, name = query.get("job", [""])[0], Path(query.get("name", [""])[0]).name
            with JOBS_LOCK:
                job = dict(JOBS.get(job_id, {}))
            if not job or name not in ALLOWED_FILES:
                self.send_json(404, {"error": "File not found"})
                return
            target = Path(str(job["output_dir"])) / name
            if not target.is_file():
                self.send_json(404, {"error": "File not found"})
                return
            content_type = "image/png" if target.suffix == ".png" else "application/json"
            self.send_bytes(200, content_type, target.read_bytes())
            return
        self.send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        try:
            payload = self.read_payload()
            if self.path == "/api/chat":
                session = str(payload.get("session") or uuid.uuid4().hex)
                reply = COORDINATOR.chat(session, str(payload.get("message", "")), bool(payload.get("data")))
                if reply["action"] == "predict":
                    self.send_json(202, {**reply, "job": start_job(payload, "saved", session)})
                else:
                    self.send_json(200, {
                        **reply,
                        "training": {
                            "targetAccuracy": project.load_config()["training"]["default_target_accuracy"],
                            "maxEpochs": project.load_config()["training"]["default_max_epochs"],
                        } if reply["action"] == "train" else None,
                    })
                return
            if self.path == "/api/run":
                mode = str(payload.get("mode", "saved"))
                if mode not in {"saved", "full"}:
                    raise ValueError("Choose Final Output or Full Process")
                if mode == "saved" and not COORDINATOR.status()["model"]:
                    self.send_json(409, {"error": "No trained model was found. Run Full Process first."})
                    return
                session = str(payload.get("session") or uuid.uuid4().hex)
                self.send_json(202, {"job": start_job(payload, mode, session)})
                return
            if self.path == "/api/llm/config":
                client = COORDINATOR.llm
                previous = (client.provider, client.endpoint, client.model, client.api_key)
                try:
                    configuration = COORDINATOR.configure_llm(payload)
                    message = client.test()
                except Exception:
                    client.provider, client.endpoint, client.model, client.api_key = previous
                    raise
                self.send_json(200, {"configuration": configuration, "message": message})
                return
            self.send_json(404, {"error": "Not found"})
        except (ValueError, TypeError, json.JSONDecodeError, binascii.Error, OSError, IndexError, KeyError) as exc:
            self.send_json(400, {"error": str(exc)})


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 8000), Handler)
    print("AgenticDerma platform: http://127.0.0.1:8000", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
