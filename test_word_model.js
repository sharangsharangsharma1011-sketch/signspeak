(function () {
  const video = document.getElementById('video');
  const canvas = document.getElementById('overlay');
  const ctx = canvas.getContext('2d');
  const startOverlay = document.getElementById('startOverlay');
  const startBtn = document.getElementById('startBtn');
  const recordBtn = document.getElementById('recordBtn');
  const statusText = document.getElementById('statusText');
  const statusDot = document.getElementById('statusDot');
  const resultsEl = document.getElementById('results');
  const bufferInfo = document.getElementById('bufferInfo');

  let weights = null;
  let recording = false;
  let recordedFrames = []; // each frame: 63-length normalized vector, or null if no hand

  function resizeCanvas() {
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
  }
  window.addEventListener('resize', resizeCanvas);

  function setStatus(live, text) {
    statusText.textContent = text;
    statusDot.classList.toggle('live', live);
  }

  // Mirrors preprocess_dataset.py: wrist-relative, scaled by wrist->middle_MCP distance
  function normalizeLandmarks(lm) {
    const wrist = lm[0];
    const pts = lm.map(p => [p.x - wrist.x, p.y - wrist.y, p.z - wrist.z]);
    const mid = pts[9];
    const scale = Math.hypot(mid[0], mid[1], mid[2]) || 1.0;
    const flat = [];
    for (const p of pts) {
      flat.push(p[0] / scale, p[1] / scale, p[2] / scale);
    }
    return flat; // length 63
  }

  // Mirrors resample_sequence() in preprocess_dataset.py: linear interpolation to fixed length
  function resampleSequence(vectors, targetLen) {
    const T = vectors.length;
    if (T === 0) return null;
    if (T === 1) return Array.from({ length: targetLen }, () => vectors[0].slice());

    const dim = vectors[0].length;
    const origIdx = Array.from({ length: T }, (_, i) => i);
    const targetIdx = Array.from({ length: targetLen }, (_, i) => (i * (T - 1)) / (targetLen - 1));

    const out = [];
    for (let ti = 0; ti < targetLen; ti++) {
      const t = targetIdx[ti];
      const lo = Math.floor(t);
      const hi = Math.min(lo + 1, T - 1);
      const frac = t - lo;
      const row = new Array(dim);
      for (let d = 0; d < dim; d++) {
        row[d] = vectors[lo][d] * (1 - frac) + vectors[hi][d] * frac;
      }
      out.push(row);
    }
    return out;
  }

  function onResults(results) {
    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const hasHand = results.multiHandLandmarks && results.multiHandLandmarks.length > 0;

    if (hasHand) {
      const lm = results.multiHandLandmarks[0];
      if (window.drawConnectors) {
        window.drawConnectors(ctx, lm, window.HAND_CONNECTIONS, { color: 'rgba(58,170,146,0.55)', lineWidth: 2 });
        window.drawLandmarks(ctx, lm, { color: 'rgba(200,200,204,0.85)', lineWidth: 0, radius: 3 });
      }

      if (recording) {
        recordedFrames.push(normalizeLandmarks(lm));
        bufferInfo.textContent = `Recording... ${recordedFrames.length} frames captured`;
      }
    } else if (recording) {
      bufferInfo.textContent = `Recording... ${recordedFrames.length} frames captured (no hand this frame)`;
    }

    ctx.restore();
  }

  recordBtn.addEventListener('click', () => {
    if (!recording) {
      recording = true;
      recordedFrames = [];
      recordBtn.textContent = 'Stop & Predict';
      recordBtn.classList.add('active');
      resultsEl.innerHTML = '';
      bufferInfo.textContent = 'Recording... 0 frames captured';
    } else {
      recording = false;
      recordBtn.textContent = 'Start Recording';
      recordBtn.classList.remove('active');
      runPrediction();
    }
  });

  function runPrediction() {
    if (recordedFrames.length < 3) {
      bufferInfo.textContent = `Only ${recordedFrames.length} frames with a hand detected — too short, try again with a slower, clearer sign.`;
      return;
    }
    const resampled = resampleSequence(recordedFrames, weights.seq_len);
    const probs = window.SignWordModel.predict(resampled, weights);

    const ranked = weights.classes
      .map((cls, i) => ({ cls, prob: probs[i] }))
      .sort((a, b) => b.prob - a.prob)
      .slice(0, 5);

    bufferInfo.textContent = `Captured ${recordedFrames.length} frames with a hand, resampled to ${weights.seq_len}.`;
    resultsEl.innerHTML = ranked.map((r, idx) => `
      <div class="result-row ${idx === 0 ? 'top' : ''}">
        <span class="result-word">${r.cls}</span>
        <div class="result-bar-track"><div class="result-bar" style="width:${(r.prob * 100).toFixed(1)}%"></div></div>
        <span class="result-pct">${(r.prob * 100).toFixed(1)}%</span>
      </div>
    `).join('');
  }

  async function loadWeights() {
    const res = await fetch('model_weights.json');
    weights = await res.json();
    console.log('Loaded model weights. Classes:', weights.classes);
  }

  async function start() {
    setStatus(false, 'Starting…');
    startBtn.disabled = true;
    startBtn.textContent = 'Starting…';

    try {
      await loadWeights();

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      video.srcObject = stream;
      await video.play();
      resizeCanvas();

      const hands = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
      });
      hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.6,
        minTrackingConfidence: 0.5,
      });
      hands.onResults(onResults);

      const camera = new Camera(video, {
        onFrame: async () => { await hands.send({ image: video }); },
        width: 640,
        height: 480,
      });
      camera.start();

      startOverlay.classList.add('hidden');
      recordBtn.disabled = false;
      setStatus(true, 'Live');
    } catch (err) {
      console.error(err);
      setStatus(false, 'Error');
      startBtn.disabled = false;
      startBtn.textContent = 'Start camera';
      alert('Could not start: ' + err.message);
    }
  }

  startBtn.addEventListener('click', start);
})();
