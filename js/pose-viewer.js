/* CapAI — pose skeleton overlay + readable feedback (vanilla JS, no framework)
 *
 * USAGE
 *   1. Drop this file at  js/pose-viewer.js
 *   2. In index.html, after your upload <script>, add:
 *        <script src="js/pose-viewer.js"></script>
 *   3. When your /analyze fetch resolves, call:
 *        CapAIPose.render(resultJson, theFileObject, document.getElementById("results"));
 *      where:
 *        - resultJson      = the parsed JSON from the backend (the { ok:true, ... } object)
 *        - theFileObject   = the File the user picked (from your <input type="file">)
 *        - the 3rd arg     = any container element to render into (it is cleared first)
 */
(function (global) {
  "use strict";

  // ── MediaPipe Pose 33-landmark skeleton edges ────────────────────────────
  var CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],[9,10],[11,12],
    [11,13],[13,15],[12,14],[14,16],[15,17],[15,19],[15,21],[17,19],
    [16,18],[16,20],[16,22],[18,20],[11,23],[12,24],[23,24],
    [23,25],[25,27],[27,29],[27,31],[29,31],
    [24,26],[26,28],[28,30],[28,32],[30,32]
  ];

  var LEFT_COLOR = "#00e5ff", RIGHT_COLOR = "#ff4081", CENTER_COLOR = "#ffffff";
  var LEFT = {1:1,2:1,3:1,7:1,9:1,11:1,13:1,15:1,17:1,19:1,21:1,23:1,25:1,27:1,29:1,31:1};
  var RIGHT = {4:1,5:1,6:1,8:1,10:1,12:1,14:1,16:1,18:1,20:1,22:1,24:1,26:1,28:1,30:1,32:1};

  function jointColor(i) { return LEFT[i] ? LEFT_COLOR : RIGHT[i] ? RIGHT_COLOR : CENTER_COLOR; }
  function edgeColor(a, b) {
    if (LEFT[a] && LEFT[b]) return LEFT_COLOR;
    if (RIGHT[a] && RIGHT[b]) return RIGHT_COLOR;
    return CENTER_COLOR;
  }

  // ── Coaching feedback ────────────────────────────────────────────────────
  function elbow(angle, side) {
    var s = side + " elbow";
    if (angle < 90)  return { label: s, note: angle + "° — very bent, arm is loaded",   color: "#ffd740" };
    if (angle < 130) return { label: s, note: angle + "° — good pulling position",       color: "#69f0ae" };
    if (angle < 160) return { label: s, note: angle + "° — arm extending",                color: "#69f0ae" };
    return                   { label: s, note: angle + "° — arm nearly straight",         color: "#aaaaaa" };
  }
  function knee(angle, side) {
    var s = side + " knee";
    if (angle < 100) return { label: s, note: angle + "° — deep bend, good drive",  color: "#69f0ae" };
    if (angle < 140) return { label: s, note: angle + "° — moderate kick bend",     color: "#69f0ae" };
    return                   { label: s, note: angle + "° — leg extended",          color: "#aaaaaa" };
  }
  function tilt(angle) {
    var a = Math.abs(angle), s = "Shoulder tilt";
    if (a < 5)  return { label: s, note: angle + "° — level shoulders",         color: "#69f0ae" };
    if (a < 15) return { label: s, note: angle + "° — slight lean",             color: "#ffd740" };
    return             { label: s, note: angle + "° — strong rotation or lean", color: "#ff6e40" };
  }

  // ── Canvas drawing ───────────────────────────────────────────────────────
  function drawSkeleton(ctx, lms, w, h) {
    var i, a, b;
    for (i = 0; i < CONNECTIONS.length; i++) {
      a = lms[CONNECTIONS[i][0]]; b = lms[CONNECTIONS[i][1]];
      if (!a || !b || a[2] < 0.2 || b[2] < 0.2) continue;
      ctx.beginPath();
      ctx.moveTo(a[0] * w, a[1] * h);
      ctx.lineTo(b[0] * w, b[1] * h);
      ctx.strokeStyle = edgeColor(CONNECTIONS[i][0], CONNECTIONS[i][1]);
      ctx.lineWidth = 3;
      ctx.globalAlpha = Math.min(a[2], b[2]);
      ctx.stroke();
    }
    for (i = 0; i < lms.length; i++) {
      if (lms[i][2] < 0.2) continue;
      ctx.beginPath();
      ctx.arc(lms[i][0] * w, lms[i][1] * h, 5, 0, Math.PI * 2);
      ctx.fillStyle = jointColor(i);
      ctx.globalAlpha = lms[i][2];
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  // ── Ball overlay (yellow box) ────────────────────────────────────────────
  function drawBalls(ctx, balls, w, h) {
    if (!balls || !balls.length) return;
    var maxWH = Math.max(w, h);
    for (var i = 0; i < balls.length; i++) {
      var b = balls[i];
      var cx = b.x * w, cy = b.y * h, r = b.r * maxWH;
      var pad = r * 2.5;
      ctx.save();
      ctx.strokeStyle = "#ffd400";
      ctx.lineWidth = 3;
      ctx.shadowColor = "#ffd400";
      ctx.shadowBlur = 8;
      ctx.strokeRect(cx - pad, cy - pad, pad * 2, pad * 2);
      ctx.shadowBlur = 0;
      ctx.fillStyle = "#ffd400";
      ctx.font = "bold 12px sans-serif";
      ctx.fillText("BALL", cx - pad, cy - pad - 5);
      ctx.restore();
    }
  }

  // ── Small DOM helpers ────────────────────────────────────────────────────
  function el(tag, css, text) {
    var n = document.createElement(tag);
    if (css) n.style.cssText = css;
    if (text != null) n.textContent = text;
    return n;
  }

  function card(label, note, color) {
    var c = el("div", "background:rgba(255,255,255,0.06);border:1px solid " + color +
      "44;border-radius:8px;padding:10px 14px;");
    c.appendChild(el("div", "font-size:11px;opacity:0.6;margin-bottom:4px;", label));
    c.appendChild(el("div", "font-size:13px;color:" + color + ";", note));
    return c;
  }

  // ── Public render() ──────────────────────────────────────────────────────
  function render(result, mediaFile, container) {
    if (!result || !result.ok || !result.frames || !result.frames.length) {
      container.textContent = "No pose detected in this file.";
      return;
    }
    container.innerHTML = "";
    container.style.cssText = "display:flex;flex-direction:column;gap:16px;color:#fff;font-family:sans-serif;";

    var isVideo = (result.source.type || "").indexOf("video/") === 0;
    var url = URL.createObjectURL(mediaFile);
    var W = result.source.width || 640, H = result.source.height || 480;
    var idx = 0;

    // Canvas wrapper
    var wrap = el("div", "position:relative;background:#000;border-radius:8px;overflow:hidden;");
    var canvas = el("canvas", "width:100%;display:block;");
    canvas.width = W; canvas.height = H;
    wrap.appendChild(canvas);

    var legend = el("div", "position:absolute;top:8px;left:8px;display:flex;gap:12px;font-size:11px;");
    legend.innerHTML =
      '<span style="color:' + LEFT_COLOR + '">■</span> Left ' +
      '<span style="color:' + RIGHT_COLOR + '">■</span> Right ' +
      '<span style="color:' + CENTER_COLOR + '">■</span> Center';
    wrap.appendChild(legend);

    var counter = el("div", "position:absolute;top:8px;right:8px;font-size:11px;opacity:0.7;");
    wrap.appendChild(counter);
    container.appendChild(wrap);

    var ctx = canvas.getContext("2d");
    var media = isVideo ? document.createElement("video") : new Image();
    media.src = url;
    if (isVideo) { media.muted = true; media.preload = "auto"; }

    var feedbackRow = el("div",
      "display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px;");
    container.appendChild(feedbackRow);

    function paint() {
      var f = result.frames[idx];
      ctx.clearRect(0, 0, W, H);
      try { ctx.drawImage(media, 0, 0, W, H); } catch (e) {}
      drawSkeleton(ctx, f.landmarks, W, H);
      drawBalls(ctx, f.balls, W, H);
      counter.textContent = "Frame " + (idx + 1) + " / " + result.frames.length;

      feedbackRow.innerHTML = "";
      feedbackRow.appendChild(card.apply(null, valuesOf(elbow(f.angles.elbow_l, "Left"))));
      feedbackRow.appendChild(card.apply(null, valuesOf(elbow(f.angles.elbow_r, "Right"))));
      feedbackRow.appendChild(card.apply(null, valuesOf(knee(f.angles.knee_l, "Left"))));
      feedbackRow.appendChild(card.apply(null, valuesOf(knee(f.angles.knee_r, "Right"))));
      feedbackRow.appendChild(card.apply(null, valuesOf(tilt(f.angles.shoulder_tilt))));
      feedbackRow.appendChild(qualityCard(f.visibility));
    }
    function valuesOf(o) { return [o.label, o.note, o.color]; }

    function qualityCard(v) {
      var total = v.tracked + v.partial + v.estimated || 1;
      var c = el("div", "background:rgba(255,255,255,0.06);border:1px solid #ffffff22;border-radius:8px;padding:10px 14px;");
      c.appendChild(el("div", "font-size:11px;opacity:0.6;margin-bottom:6px;", "Tracking quality"));
      var row = el("div", "display:flex;gap:6px;flex-wrap:wrap;");
      row.appendChild(el("span", "font-size:12px;color:#69f0ae;", "● " + v.tracked + " tracked"));
      row.appendChild(el("span", "font-size:12px;color:#ffd740;", "● " + v.partial + " partial"));
      row.appendChild(el("span", "font-size:12px;color:#aaa;", "● " + v.estimated + " estimated"));
      c.appendChild(row);
      var bar = el("div", "margin-top:8px;height:4px;border-radius:2px;background:#333;overflow:hidden;display:flex;");
      bar.appendChild(el("div", "width:" + (v.tracked / total * 100) + "%;background:#69f0ae;"));
      bar.appendChild(el("div", "width:" + (v.partial / total * 100) + "%;background:#ffd740;"));
      bar.appendChild(el("div", "width:" + (v.estimated / total * 100) + "%;background:#555;"));
      c.appendChild(bar);
      return c;
    }

    // Frame scrubber for videos
    if (result.frames.length > 1) {
      var scrub = el("div", "display:flex;align-items:center;gap:8px;");
      scrub.appendChild(el("span", "font-size:12px;opacity:0.6;white-space:nowrap;", "Frame"));
      var slider = el("input", "flex:1;accent-color:#00e5ff;");
      slider.type = "range"; slider.min = 0; slider.max = result.frames.length - 1; slider.value = 0;
      slider.addEventListener("input", function () {
        idx = parseInt(slider.value, 10);
        if (isVideo) { media.currentTime = result.frames[idx].index / 30; } else { paint(); }
      });
      scrub.appendChild(slider);
      container.insertBefore(scrub, feedbackRow);
    }

    if (isVideo) {
      media.addEventListener("loadeddata", function () { media.currentTime = result.frames[0].index / 30; });
      media.addEventListener("seeked", paint);
    } else {
      media.addEventListener("load", paint);
    }
  }

  global.CapAIPose = { render: render };
})(window);
