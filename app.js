(function () {
  const video = document.getElementById('video');
  const canvas = document.getElementById('overlay');
  const ctx = canvas.getContext('2d');
  const startOverlay = document.getElementById('startOverlay');
  const startBtn = document.getElementById('startBtn');
  const statusDot = document.getElementById('statusDot');
  const statusText = document.getElementById('statusText');
  const handCountEl = document.getElementById('handCount');
  const letterBadge = document.getElementById('letterBadge');
  const currentLetterEl = document.getElementById('currentLetter');
  const currentConfEl = document.getElementById('currentConf');
  const wordText = document.getElementById('wordText');
  const spaceBtn = document.getElementById('spaceBtn');
  const backspaceBtn = document.getElementById('backspaceBtn');
  const clearBtn = document.getElementById('clearBtn');
  const skeletonToggle = document.getElementById('skeletonToggle');

  let showSkeleton = true;
  let word = '';
  let heldLetter = null;
  let heldSince = 0;
  let lastCommitted = null;
  const HOLD_MS = 700; // how long a letter must be steady before it's added

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

  function onResults(results) {
    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const hasHand = results.multiHandLandmarks && results.multiHandLandmarks.length > 0;
    handCountEl.classList.toggle('hidden', false);
    handCountEl.classList.toggle('active', hasHand);
    handCountEl.textContent = hasHand ? 'Hand detected' : 'No hand detected';

    if (hasHand) {
      const lm = results.multiHandLandmarks[0];

      if (showSkeleton && window.drawConnectors) {
        window.drawConnectors(ctx, lm, window.HAND_CONNECTIONS, { color: 'rgba(58,170,146,0.55)', lineWidth: 2 });
        window.drawLandmarks(ctx, lm, { color: 'rgba(200,200,204,0.85)', lineWidth: 0, radius: 3 });
      }

      const result = window.SignClassifier.classifyLetter(lm);
      letterBadge.classList.remove('hidden');

      if (result.letter) {
        currentLetterEl.textContent = result.letter;
        currentConfEl.textContent = result.conf > 0.6 ? 'confident' : 'holding';

        const now = performance.now();
        if (result.letter === heldLetter) {
          const elapsed = now - heldSince;
          if (elapsed > HOLD_MS && lastCommitted !== result.letter) {
            commitLetter(result.letter);
            lastCommitted = result.letter;
          }
        } else {
          heldLetter = result.letter;
          heldSince = now;
          lastCommitted = null;
        }
      } else {
        currentLetterEl.textContent = '–';
        currentConfEl.textContent = 'unclear';
        heldLetter = null;
        lastCommitted = null;
      }
    } else {
      letterBadge.classList.add('hidden');
      heldLetter = null;
      lastCommitted = null;
    }

    ctx.restore();
  }

  function commitLetter(letter) {
    if (letter === '5') return; // not a real letter, just open-hand rest state
    word += letter;
    wordText.textContent = word;
    if (navigator.vibrate) navigator.vibrate(15);
  }

  spaceBtn.addEventListener('click', () => {
    word += ' ';
    wordText.textContent = word;
  });
  backspaceBtn.addEventListener('click', () => {
    word = word.slice(0, -1);
    wordText.textContent = word;
  });
  clearBtn.addEventListener('click', () => {
    word = '';
    wordText.textContent = word;
  });
  skeletonToggle.addEventListener('click', () => {
    showSkeleton = !showSkeleton;
    skeletonToggle.classList.toggle('on', showSkeleton);
  });
  skeletonToggle.classList.add('on');

  async function start() {
    setStatus(false, 'Starting…');
    startBtn.disabled = true;
    startBtn.textContent = 'Starting…';

    try {
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
        onFrame: async () => {
          await hands.send({ image: video });
        },
        width: 640,
        height: 480,
      });
      camera.start();

      startOverlay.classList.add('hidden');
      setStatus(true, 'Live');
    } catch (err) {
      console.error(err);
      setStatus(false, 'Camera error');
      startBtn.disabled = false;
      startBtn.textContent = 'Start camera';
      alert('Could not access camera. Please allow camera permission and try again.');
    }
  }

  startBtn.addEventListener('click', start);
})();
