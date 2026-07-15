// Rule-based ASL fingerspelling classifier.
// Works on 21 MediaPipe hand landmarks (normalized x,y,z).
// Covers static letters (A-Y minus J which requires motion).
// This is geometry-based, not ML-based — good enough for V1, swappable later.

const LM = {
  WRIST: 0,
  THUMB_CMC: 1, THUMB_MCP: 2, THUMB_IP: 3, THUMB_TIP: 4,
  INDEX_MCP: 5, INDEX_PIP: 6, INDEX_DIP: 7, INDEX_TIP: 8,
  MIDDLE_MCP: 9, MIDDLE_PIP: 10, MIDDLE_DIP: 11, MIDDLE_TIP: 12,
  RING_MCP: 13, RING_PIP: 14, RING_DIP: 15, RING_TIP: 16,
  PINKY_MCP: 17, PINKY_PIP: 18, PINKY_DIP: 19, PINKY_TIP: 20,
};

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0));
}

// Returns true if a finger is extended, by comparing tip distance from wrist
// against pip distance from wrist (extended fingers reach further).
function isExtended(lm, mcp, pip, tip) {
  const wrist = lm[LM.WRIST];
  return dist(wrist, lm[tip]) > dist(wrist, lm[pip]) * 1.15 &&
         dist(wrist, lm[tip]) > dist(wrist, lm[mcp]) * 1.3;
}

function isCurled(lm, mcp, pip, tip) {
  const wrist = lm[LM.WRIST];
  return dist(wrist, lm[tip]) < dist(wrist, lm[pip]) * 1.05;
}

function thumbExtended(lm) {
  // Thumb extended sideways away from palm (distance from index MCP to thumb tip is large)
  const palmWidth = dist(lm[LM.INDEX_MCP], lm[LM.PINKY_MCP]);
  const thumbSpread = dist(lm[LM.THUMB_TIP], lm[LM.INDEX_MCP]);
  return thumbSpread > palmWidth * 0.75;
}

function fingerState(lm) {
  return {
    thumb: thumbExtended(lm),
    index: isExtended(lm, LM.INDEX_MCP, LM.INDEX_PIP, LM.INDEX_TIP),
    middle: isExtended(lm, LM.MIDDLE_MCP, LM.MIDDLE_PIP, LM.MIDDLE_TIP),
    ring: isExtended(lm, LM.RING_MCP, LM.RING_PIP, LM.RING_TIP),
    pinky: isExtended(lm, LM.PINKY_MCP, LM.PINKY_PIP, LM.PINKY_TIP),
  };
}

function curl(lm, mcp, pip, tip) {
  return isCurled(lm, mcp, pip, tip);
}

// Classify a single frame's landmarks into a letter guess + rough confidence.
function classifyLetter(lm) {
  const f = fingerState(lm);
  const extCount = [f.index, f.middle, f.ring, f.pinky].filter(Boolean).length;

  const indexCurled = curl(lm, LM.INDEX_MCP, LM.INDEX_PIP, LM.INDEX_TIP);
  const middleCurled = curl(lm, LM.MIDDLE_MCP, LM.MIDDLE_PIP, LM.MIDDLE_TIP);
  const ringCurled = curl(lm, LM.RING_MCP, LM.RING_PIP, LM.RING_TIP);
  const pinkyCurled = curl(lm, LM.PINKY_MCP, LM.PINKY_PIP, LM.PINKY_TIP);

  const thumbIndexDist = dist(lm[LM.THUMB_TIP], lm[LM.INDEX_TIP]);
  const thumbMiddleDist = dist(lm[LM.THUMB_TIP], lm[LM.MIDDLE_TIP]);
  const palmWidth = dist(lm[LM.INDEX_MCP], lm[LM.PINKY_MCP]);

  // B: four fingers extended straight up, thumb tucked across palm
  if (f.index && f.middle && f.ring && f.pinky && !f.thumb) return { letter: 'B', conf: 0.8 };

  // A: fist with thumb alongside (not crossed over fingers), all curled
  if (indexCurled && middleCurled && ringCurled && pinkyCurled && f.thumb) return { letter: 'A', conf: 0.7 };

  // S: tight fist, thumb across front of curled fingers
  if (indexCurled && middleCurled && ringCurled && pinkyCurled && !f.thumb) return { letter: 'S', conf: 0.65 };

  // D: index up, others curled, thumb touches middle finger
  if (f.index && middleCurled && ringCurled && pinkyCurled && thumbMiddleDist < palmWidth * 0.5) return { letter: 'D', conf: 0.7 };

  // I: pinky up only
  if (!f.index && !f.middle && !f.ring && f.pinky) return { letter: 'I', conf: 0.75 };

  // L: index + thumb extended in an L shape, others curled
  if (f.index && !f.middle && !f.ring && !f.pinky && f.thumb) return { letter: 'L', conf: 0.75 };

  // Y: thumb + pinky extended, others curled ("hang loose")
  if (!f.index && !f.middle && !f.ring && f.pinky && f.thumb) return { letter: 'Y', conf: 0.75 };

  // V / U / K: index + middle extended
  if (f.index && f.middle && ringCurled && pinkyCurled) {
    const spread = dist(lm[LM.INDEX_TIP], lm[LM.MIDDLE_TIP]);
    if (f.thumb && spread > palmWidth * 0.5) return { letter: 'K', conf: 0.6 };
    if (spread > palmWidth * 0.45) return { letter: 'V', conf: 0.7 };
    return { letter: 'U', conf: 0.65 };
  }

  // W: index, middle, ring extended, pinky curled
  if (f.index && f.middle && f.ring && pinkyCurled) return { letter: 'W', conf: 0.7 };

  // F: thumb + index touching (circle), other three extended
  if (thumbIndexDist < palmWidth * 0.35 && f.middle && f.ring && f.pinky) return { letter: 'F', conf: 0.65 };

  // O: all fingertips curl toward thumb tip, forming a circle
  if (thumbIndexDist < palmWidth * 0.4 && indexCurled && middleCurled && ringCurled && pinkyCurled) return { letter: 'O', conf: 0.6 };

  // C: curved hand, fingers together and bent, not fully closed
  if (extCount === 0 && !indexCurled && thumbIndexDist > palmWidth * 0.4 && thumbIndexDist < palmWidth * 0.9) {
    return { letter: 'C', conf: 0.5 };
  }

  // E: all fingertips curled tightly toward palm, thumb across
  if (indexCurled && middleCurled && ringCurled && pinkyCurled && thumbIndexDist < palmWidth * 0.3) {
    return { letter: 'E', conf: 0.55 };
  }

  // G/Q: index extended pointing sideways, thumb extended, others curled (approximate as G)
  if (f.index && !f.middle && !f.ring && !f.pinky && !f.thumb) return { letter: 'G', conf: 0.55 };

  // H: index + middle extended sideways together, ring/pinky curled, thumb tucked (approximate, overlaps U)
  // handled by U above in most cases

  // M/N: thumb tucked under curled fingers — approximate as low-confidence M
  if (indexCurled && middleCurled && ringCurled && pinkyCurled && f.thumb === false && thumbIndexDist < palmWidth * 0.55) {
    return { letter: 'M', conf: 0.4 };
  }

  // R: index + middle crossed
  // T, X, N, Q, P, Z are harder to distinguish geometrically without more precision — omitted from V1 confident set

  // Open hand, all extended + thumb: fallback letter 5-hand shape (no direct ASL letter, but useful as "open")
  if (f.thumb && f.index && f.middle && f.ring && f.pinky) return { letter: '5', conf: 0.3 };

  return { letter: null, conf: 0 };
}

window.SignClassifier = { classifyLetter };
