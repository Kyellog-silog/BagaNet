export const isBrowser = typeof window !== 'undefined';
export const API_CONFIG = { BASE_URL: 'http://localhost:5050', API_KEY: 'FeDMl2025' };
import * as tf from '@tensorflow/tfjs';
import pako from 'pako';

// Helper to get a clean JWT (or null)
export function getJwtToken() {
    const token = localStorage.getItem('jwtToken');
    return token && token !== 'null' ? token : null;
}

/**
 * Manually trigger purging of images marked as used
 * @param {Function} progressCallback - Optional callback for purge progress
 * @returns {Promise<Object>} - Results of the purge operation
 */
export async function manualPurgeUsedImages(progressCallback = () => {}) {
    try {
      progressCallback(0);
  
      // Get all images marked as used
      const usedImages = await getImagesByTrainingStatus('used');
      if (!usedImages || usedImages.length === 0) {
        progressCallback(1);
        return { 
          success: true, 
          message: "No used images found to purge", 
          purgedCount: 0 
        };
      }
  
      progressCallback(0.3);
      const imageIds = usedImages.map(img => img.id);
      const result = await purgeUsedImages();
      progressCallback(0.8);
  
      console.log(`Successfully purged ${result.purgedCount} used images`);
      progressCallback(1);
  
      return {
        success: true,
        message: `Successfully purged ${result.purgedCount} images`,
        purgedCount: result.purgedCount,
        purgedIds: imageIds
      };
    } catch (error) {
      console.error("Error during manual purge:", error);
      return {
        success: false,
        message: `Error purging images: ${error.message}`,
        error: error.toString()
      };
    }
  }



export async function fetchTfjsModel(progressCallback = () => {}) {
  const tf = await import('@tensorflow/tfjs');

    console.log('io.js model initialized');
  // Configure memory settings before any operations
  tf.ENV.set('WEBGL_DELETE_TEXTURE_THRESHOLD', 0); // Aggressive cleanup
  tf.ENV.set('WEBGL_PACK', true); // Enable texture packing
  tf.ENV.set('WEBGL_FORCE_F16_TEXTURES', true); // Use 16-bit floats
  tf.ENV.set('WEBGL_FLUSH_THRESHOLD', 1); // More frequent flushing


  const defaultParams = {
    epochs: 8,
    batch_size: 8, // Reduced from 32 to lower memory pressure
    learning_rate: 0.001,
    min_local_samples: 2
  };

  // Memory monitoring
  const initialMemory = tf.memory().numBytes;

  // 1. Try loading progressively smaller models
  const modelAttempts = [
    { key: 'indexeddb://tfjs_int8', type: 'int8' },
    { key: 'indexeddb://tfjs_full', type: 'full' },
    { key: 'indexeddb://tfjs_default', type: 'default' }
  ];

  for (const attempt of modelAttempts) {
    try {
      console.log(`🔄 Attempting to load ${attempt.type} model...`);
      const model = await tf.tidy(() => 
        tf.loadLayersModel(attempt.key)
      );
      
      // Clone weights with memory management
      const weights = await tf.tidy(() => 
        model.getWeights().map(w => w.clone())
      );
      
      console.log(`✅ Loaded ${attempt.type} model with ${weights.length} weights`);
      console.log('Memory delta:', (tf.memory().numBytes - initialMemory) / (1024 * 1024), 'MB');
      
      return {
        model,
        trainingParams: {
          ...defaultParams,
          batch_size: attempt.type === 'int8' ? 4 : 8 // Adjust batch size based on model type
        },
        modelType: attempt.type
      };
    } catch (err) {
      console.warn(`⚠️ Could not load ${attempt.type} model:`, err.message);
      tf.disposeVariables(); // Clean up any partial loads
    }
  }

  // 2. Fetch from server if local loading failed
  console.log('🔄 Fetching model from server...');
  try {
    const res = await fetch(`${API_CONFIG.BASE_URL}/tfjs/training_model`, {
      method: 'GET',
      headers: {
        'X-API-Key': API_CONFIG.API_KEY,
        'X-Client-ID': localStorage.getItem('clientId')
      }
    });
    
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    
    const data = await res.json();
    
    // Load with memory management
    const model = await tf.tidy(async () => {
      const m = await tf.loadLayersModel(data.model_url);
      
      // Process weights in chunks to avoid memory spikes
      const weights = [];
      const weightChunks = [];
      const allWeights = m.getWeights();
      
      for (let i = 0; i < allWeights.length; i += 2) { // Process 2 weights at a time
        const chunk = allWeights.slice(i, i + 2);
        const cloned = await tf.tidy(() => chunk.map(w => w.clone()));
        weights.push(...cloned);
        tf.dispose(chunk);
        weightChunks.push(cloned);
      }
      
      return { model: m, weights };
    });

    // Save to IndexedDB in a memory-efficient way
    await tf.tidy(() => model.model.save('indexeddb://tfjs_default'));
    
    console.log('✅ Fetched and saved server model');
    console.log('Memory delta:', (tf.memory().numBytes - initialMemory) / (1024 * 1024), 'MB');

    return {
      model: model.model,
      trainingParams: data.training_params || defaultParams,
      modelType: 'server',
      weights: model.weights
    };
  } catch (err) {
    console.error('❌ Failed to fetch TFJS model:', err);
    tf.disposeVariables();
    throw err;
  }
}

/** 
 * Creates a typed array view into an ArrayBuffer 
 * @param {'float32'|'int32'|'bool'} dtype 
 * @param {ArrayBuffer} buffer 
 * @param {number} offsetElems  // offset in ELEMENTS, not bytes
 * @param {number} lengthElems  // number of elements
 */
function getTypedArrayForDType(dtype, buffer, offsetElems, lengthElems) {
    const byteOffset = offsetElems * bytesPerElement(dtype);
    switch (dtype) {
      case 'float32': return new Float32Array(buffer, byteOffset, lengthElems);
      case 'int32':   return new Int32Array(buffer, byteOffset, lengthElems);
      case 'bool':    return new Uint8Array(buffer, byteOffset, lengthElems);
      default: throw new Error(`Unsupported dtype ${dtype}`);
    }
}
export async function submitTrainedModel(trainedModel, systemMetrics = {}, progressCallback = () => {}) {
    try {
        // 1. Verify model exists
        if (!trainedModel?.model) {
            throw new Error("Invalid model object");
        }

        progressCallback(10);
        console.log("[1] Starting model save...");

        // 2. Save model and get artifacts
        const artifacts = await trainedModel.model.save(tf.io.withSaveHandler(async (artifacts) => {
            console.log("[2] Inside withSaveHandler");
            progressCallback(30);
            return artifacts;
        }));
        console.log("[3] Model save complete:", artifacts);

        // 3. Prepare payload
        progressCallback(50);
        const payload = {
            metrics: {
                accuracy:    Number(trainedModel.accuracy?.toFixed(4)) || 0,
                loss:        Number(trainedModel.loss?.toFixed(4))     || 0,
                val_accuracy:Number(trainedModel.valAccuracy?.toFixed(4)) || 0,
                val_loss:    Number(trainedModel.valLoss?.toFixed(4)) || 0,
                precision:   Number(trainedModel.precision?.toFixed(4)) || 0,
                recall:      Number(trainedModel.recall?.toFixed(4))    || 0,
                f1_score:    Number(trainedModel.f1Score?.toFixed(4))  || 0
            },
            dataset_stats: {
                train_size: trainedModel.trainSize || 0,
                val_size: trainedModel.valSize || 0,
                classes: trainedModel.classes || []
            },
            system_metrics: systemMetrics,
            timestamp: new Date().toISOString()
        };
        console.log("[4] Payload prepared:", payload);

        // 4. Extract and process weights
        progressCallback(70);
        console.log("[5] Getting weights...");
        const weights = trainedModel.model.getWeights();
        const weightSpecs = [];

        let totalSize = 0;
        weights.forEach((weight, i) => {
            const shape = weight.shape;
            const size = shape.reduce((a, b) => a * b, 1);
            totalSize += size;

            weightSpecs.push({
                name: `layer_${i}`,
                shape: shape,
                dtype: 'float32'
            });
        });

        console.log("[6] Total weights size:", totalSize);

        const combinedBuffer = new Float32Array(totalSize);
        let offset = 0;
        for (const weight of weights) {
            const data = await weight.data();
            combinedBuffer.set(data, offset);
            offset += data.length;
        }

        console.log("[7] Weights combined successfully");

        // 5. Prepare form data
        const form = new FormData();
        form.append('model', new Blob([
            JSON.stringify({
                modelTopology: artifacts.modelTopology,
                weightSpecs: weightSpecs
            })
        ], { type: 'application/json' }));
        form.append('weights', new Blob([combinedBuffer.buffer], { type: 'application/octet-stream' }));
        form.append('metadata', JSON.stringify(payload));

        console.log("[8] Form data prepared:");
        for (const [key, value] of form.entries()) {
            console.log(`    • ${key}:`, value);
        }

        // 6. Submit using fetch with timeout
        progressCallback(85);
        console.log("[9] Sending fetch request...");

        const controller = new AbortController();
        const timeoutId = setTimeout(() => {
            controller.abort();
            console.error("Fetch aborted due to timeout");
        }, 180000); // 60 seconds

        const response = await fetch('http://localhost:5050/tfjs/submit_weights', {
            method: 'POST',
            headers: {
                'X-API-Key': API_CONFIG.API_KEY,
                'X-Client-ID': localStorage.getItem('clientId') || 'unknown',
                ...(getJwtToken() ? { 'Authorization': `Bearer ${getJwtToken()}` } : {}),
            },
            body: form,
            signal: controller.signal
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.message || `Server error: ${response.status}`);
        }

        progressCallback(100);
        console.log("[10] Weights submitted successfully");
        const result = await response.json();
        console.log("[11] Server response:", result);

        return result;

    } catch (error) {
        console.error("❌ Submission failed:", error.message);
        progressCallback(0);
        throw error;
    }
}
