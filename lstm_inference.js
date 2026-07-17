// Minimal LSTM + Dense forward pass in plain JS, matching the trained Keras
// model exactly (2 stacked LSTMs -> Dense(relu) -> Dense(softmax)).
// This avoids needing the full TF.js runtime for a model this small.
//
// Keras LSTM gate order in the kernel/recurrent_kernel/bias arrays is
// [input, forget, cell, output] (i, f, c, o), each of size `units`.

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
function tanh(x) { return Math.tanh(x); }

// Multiply a vector by a matrix stored as [inputDim][outputDim] (row-major, Keras layout)
function vecMatMul(vec, mat) {
  const outDim = mat[0].length;
  const out = new Array(outDim).fill(0);
  for (let i = 0; i < vec.length; i++) {
    const row = mat[i];
    for (let j = 0; j < outDim; j++) {
      out[j] += vec[i] * row[j];
    }
  }
  return out;
}

function addVec(a, b) {
  return a.map((v, i) => v + b[i]);
}

// Runs one LSTM layer over a full sequence, returns either the full output
// sequence (returnSequences=true) or just the final hidden state.
function runLSTM(inputSeq, layer, returnSequences) {
  const units = layer.units;
  let h = new Array(units).fill(0);
  let c = new Array(units).fill(0);
  const outputs = [];

  for (const xT of inputSeq) {
    const z = addVec(
      addVec(vecMatMul(xT, layer.kernel), vecMatMul(h, layer.recurrent_kernel)),
      layer.bias
    );
    const i = z.slice(0, units).map(sigmoid);
    const f = z.slice(units, 2 * units).map(sigmoid);
    const g = z.slice(2 * units, 3 * units).map(tanh);
    const o = z.slice(3 * units, 4 * units).map(sigmoid);

    c = c.map((cPrev, idx) => f[idx] * cPrev + i[idx] * g[idx]);
    h = c.map((cVal, idx) => o[idx] * tanh(cVal));

    if (returnSequences) outputs.push(h.slice());
  }

  return returnSequences ? outputs : h;
}

function denseRelu(x, layer) {
  const z = addVec(vecMatMul(x, layer.kernel), layer.bias);
  return z.map((v) => Math.max(0, v));
}

function denseSoftmax(x, layer) {
  const z = addVec(vecMatMul(x, layer.kernel), layer.bias);
  const max = Math.max(...z);
  const exps = z.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

// Full forward pass: sequence (SEQ_LEN x 63) -> class probabilities
function predict(sequence, weights) {
  const lstm1Out = runLSTM(sequence, weights.lstm1, true);
  const lstm2Out = runLSTM(lstm1Out, weights.lstm2, false);
  const dense1Out = denseRelu(lstm2Out, weights.dense1);
  // dropout is a no-op at inference time
  const probs = denseSoftmax(dense1Out, weights.dense2);
  return probs;
}

window.SignWordModel = { predict };
