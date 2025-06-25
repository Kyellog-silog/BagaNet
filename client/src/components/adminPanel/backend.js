import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';


export function initWasmBackend(pathPrefix = '/assets/wasm/') {
  tf.wasm.setWasmPaths({
    'tfjs-backend-wasm.wasm': `${pathPrefix}tfjs-backend-wasm.wasm`,
  });
}


export async function chooseBestTfBackend() {
  const candidates = ['webgl', 'wasm', 'cpu'];

  for (let name of candidates) {
    if (!tf.findBackend(name)) continue;

    try {
      await tf.setBackend(name);
      await tf.ready();
      console.log(`✅ Using TF.js backend: ${name}`);
      return name;
    } catch (e) {
      console.warn(`❌ Failed to set TF backend to "${name}":`, e);
      // skip to next
    }
  }

  throw new Error('No TensorFlow.js backend could be initialized');
}
