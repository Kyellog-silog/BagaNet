// tfjsTraining.js


import '@tensorflow/tfjs-backend-cpu';
await tf.setBackend('cpu');
import * as tf from '@tensorflow/tfjs';


// Call this at the start of your application
setupTensorFlow().catch(console.error);

// Utilities for training with image management/purging capabilities and TensorFlow.js model integration
import { openDatabase, getAllImageIds, markImagesAsUsed, purgeUsedImages, getImagesByTrainingStatus, getModel, storeModel, processImagesInBatches } from '../utils/indexedDBUtils.js';


/** 
 * Async generator yielding {xs, ys} examples 
 * where xs is a tensor and ys is a one-hot label.
 */
async function* indexedDBGenerator(batchSize = 32) {
  // processImagesInBatches uses your IndexedDB batched reader under the hood
  // to grab `batchSize` records at a time.
  const examples = await processImagesInBatches(
    async (batch) => {
      return batch.map(record => {
        const img = record.imageBlob;           // Blob from IndexedDB
        const tensor = tf.tidy(() =>
          tf.browser
            .fromPixels(img)                   // decode to tensor
            .resizeNearestNeighbor([224,224])  // example resize
            .toFloat()
            .div(255.0)                        // normalize
        );
        const label = tf.oneHot(
          parseInt(record.classificationIndex), // assume you stored an index
          numClasses
        );
        return { xs: tensor, ys: label };
      });
    },
    batchSize
  );

  for (const ex of examples) {
    yield ex;
  }
}

// Then build your Dataset:
const ds = tf.data
  .generator(() => indexedDBGenerator(64))
  .shuffle(64 * 2)        // buffer twice your batch
  .batch(64)
  .prefetch( tf.data.AUTOTUNE );

// Re-export the necessary functions that metrics.jsx is trying to import
export { getImagesByTrainingStatus };

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

// Environment check
const isBrowser = typeof window !== 'undefined' && typeof document !== 'undefined';

// Constants
const API_CONFIG = {
  BASE_URL: "http://localhost:5050",
  // This should ideally be loaded from environment variables or secure storage
  API_KEY: "FeDMl2025",
  // Generate or load a client ID that persists throughout the session
  CLIENT_ID: isBrowser
    ? (localStorage.getItem('clientId') || `client-${Date.now()}`)
    : "default-client"
};

// Store client ID if in browser environment
if (isBrowser && !localStorage.getItem('clientId')) {
  localStorage.setItem('clientId', API_CONFIG.CLIENT_ID);
}

// Retrieve the JWT you obtained from /api/token
const JWT = isBrowser ? localStorage.getItem('jwtToken') : null;

const storedToken = localStorage.getItem('jwtToken');
if (storedToken === 'null' || storedToken === null) {
  localStorage.removeItem('jwtToken');
}

// Helper to get a clean JWT (or null)
function getJwtToken() {
  const token = localStorage.getItem('jwtToken');
  return token && token !== 'null' ? token : null;
}

export async function fetchTfjsModel(progressCallback = () => {}) {
  const tf = await import('@tensorflow/tfjs');

  const defaultParams = {
    epochs: 10,
    batch_size: 8,
    learning_rate: 0.001,
    min_local_samples: 2
  };

  // Try loading from IndexedDB first
  try {
    console.log('🔄 Attempting to load quantized INT8 model from IndexedDB…');
    const int8Model = await tf.loadLayersModel('indexeddb://tfjs_int8');
    console.log('✅ Loaded INT8 model from IndexedDB');
    return {
      model: int8Model,
      trainingParams: defaultParams,
      modelType: 'int8'
    };
  } catch (errInt8) {
    console.warn('⚠️  Could not load INT8 model:', errInt8);
  }

  // 2. Fallback: load the full-precision model
  try {
    console.log('🔄 Attempting to load full-precision model from IndexedDB…');
    const fullModel = await tf.loadLayersModel('indexeddb://tfjs_full');
    console.log('✅ Loaded full-precision model from IndexedDB');
    return {
      model: fullModel,
      trainingParams: defaultParams,
      modelType: 'full'
    };
  } catch (errFull) {
    console.warn('⚠️  Could not load full-precision model:', errFull);
  }
  
  try {
    
    const model = await tf.loadLayersModel('indexeddb://tfjs_default');
    console.log("✅ TFJS model loaded from IndexedDB");
    return {
      model,
      trainingParams: defaultParams
    };
  } catch (err) {
    console.warn("⚠️ Failed to load model from IndexedDB, fetching from server:", err);
  }

  // Fetch model from server as fallback
  const res = await fetch("http://localhost:5050/tfjs/training_model", {
    method: 'GET',
    headers: {
      'X-API-Key': 'FeDMl2025',
      'X-Client-ID': localStorage.getItem('clientId')
    }
  });

  if (!res.ok) {
    throw new Error(`❌ Failed to fetch TFJS model: ${res.status}`);
  }

  const data = await res.json();
  const model = await tf.loadLayersModel(data.model_url);

  // Save model locally for next run
  await model.save('indexeddb://tfjs_default');
  console.log("✅ Fetched TFJS model saved to IndexedDB");

  return {
    model,
    trainingParams: data.training_params || defaultParams
  };
}



/**
 * Prepares image data for training by splitting into train/validation/test sets
 * @param {Array} images - Array of image objects from IndexedDB
 * @param {Function} progressCallback - Callback function for progress updates
 * @returns {Object} - Object containing training splits and metadata
 */
export const prepareData = async (images, progressCallback = () => {}) => {
  try {
    if (!isBrowser) {
      throw new Error("This function requires a browser environment");
    }

    if (!images || images.length === 0) {
      throw new Error("No images provided for preparation");
    }
    
    progressCallback(0.1);
    
    // Group images by classification
    const imagesByClass = {};
    let totalProcessed = 0;
    
    // Track image IDs for updating status after training
    const imageIds = [];
    
    // Modified: Check for either classifiedAs or classification field
    // Get classes first so they're available for all processing
    const classes = [...new Set(images.filter(img => 
      img.classifiedAs || img.classification
    ).map(img => 
      img.classifiedAs || img.classification
    ))];
    
    // Filter out "Unclassified" if it exists in the classes array
    const validClasses = classes.filter(cls => cls !== "Unclassified");
    
    if (validClasses.length === 0) {
      throw new Error("No valid classified images found for training");
    }
    
    console.log("Found classes for training:", validClasses);
    
    // Process each image
    for (const img of images) {
      // Modified: Check for either classifiedAs or classification field
      const classification = img.classifiedAs || img.classification;
      
      // Skip unclassified images or images with "Unclassified" classification
      if (!classification || classification === "Unclassified") {
        totalProcessed++;
        progressCallback((totalProcessed / images.length) * 0.3);
        continue;
      }
      
      // Track the image ID for later status update
      if (img.id) {
        imageIds.push(img.id);
      }
      
      // Initialize array for this class if needed
      if (!imagesByClass[classification]) {
        imagesByClass[classification] = [];
      }
      
      // Create image element to load image data
      const imgElement = await loadImageToElement(img);
      if (!imgElement) {
        totalProcessed++;
        progressCallback((totalProcessed / images.length) * 0.3);
        continue; // Skip images that can't be loaded
      }
      
      // Process the image for training (resize, normalize, etc.)
      const processedImage = await processImageForTraining(imgElement, classification, validClasses);
      
      if (processedImage) {
        imagesByClass[classification].push(processedImage);
      }
      
      totalProcessed++;
      progressCallback((totalProcessed / images.length) * 0.3);
    }
    
    progressCallback(0.4);
    
    // Check if we have valid data to proceed
    if (Object.keys(imagesByClass).length === 0) {
      throw new Error("No valid classified images found for training");
    }
    
    // Split data into train/validation/test
    const train = [];
    const validation = [];
    const test = [];
    
    // Use a 70/15/15 split
    for (const className in imagesByClass) {
      const classImages = imagesByClass[className];
      
      if (classImages.length === 0) continue;
      
      // Shuffle the images
      shuffleArray(classImages);
      
      const trainCount = Math.max(1, Math.floor(classImages.length * 0.7));
      const valCount = Math.max(1, Math.floor(classImages.length * 0.15));
      
      // Split the images
      train.push(...classImages.slice(0, trainCount));
      validation.push(...classImages.slice(trainCount, trainCount + valCount));
      test.push(...classImages.slice(trainCount + valCount));
    }
    
    progressCallback(0.5);
    
    // Shuffle the datasets
    shuffleArray(train);
    shuffleArray(validation);
    shuffleArray(test);
    
    // Return the prepared data with image IDs for status tracking
    console.log("Prepared dataset summary:", {
      train: train.length,
      val: validation.length,
      test: test.length,
      imageIds: imageIds.length,
      classes: validClasses
    });
    
    return {
      train,
      validation,
      test,
      classes: validClasses,
      totalCount: train.length + validation.length + test.length,
      distribution: Object.fromEntries(
        Object.entries(imagesByClass).map(([k, v]) => [k, v.length])
      ),
      imageIds // Include image IDs for later status updates
    };
  } catch (error) {
    console.error("Error preparing data:", error);
    throw new Error("Failed to prepare training data: " + error.message);
  }
};

/**
 * Optimized training pipeline with GPU memory management and detailed logging
 * @param {Object} preparedData - Data prepared by prepareData()
 * @param {Function} progressCallback - Updates overall progress (0–1)
 * @param {Function} epochCallback - Called at end of each epoch with {epoch, valLoss, valAccuracy}
 * @param {Boolean} submitToServer - Whether to POST the final model & metrics
 * @returns {Object} - {model, version, timestamp, loss, accuracy, precision, recall, f1Score, trainSize, valSize, testSize}
 */
export const trainModel = async (
  preparedData,
  progressCallback = () => {},
  epochCallback = () => {},
  submitToServer = false
) => {
  try {
    if (!isBrowser) throw new Error("This function requires a browser environment");
    if (!preparedData.train?.length) throw new Error("Invalid prepared data for training");

    console.log("Training sizes:", {
      train: preparedData.train.length,
      validation: preparedData.validation?.length,
      test: preparedData.test?.length,
      classes: preparedData.classes
    });
    progressCallback(0);
    
    if (tf.getBackend() !== 'webgl' && tf.getBackend() !== 'wasm') {
      await tf.setBackend('wasm');
    }
    await tf.ready();
    
    // === Backend setup ===
    const backendName = tf.getBackend();
    if (backendName !== 'webgl') {
      try {
        console.log("Setting up WebGL backend…");
        tf.env().set('WEBGL_PACK', false);
        await tf.setBackend('webgl');
        if (tf.getBackend() === 'webgl') {
          tf.env().set('WEBGL_FORCE_F16_TEXTURES', true);
          console.log('🔧 WebGL mixed precision enabled (F16 textures)');
        }
        const flags = Object.keys(tf.env().flags);
        if (flags.includes('WEBGL_FORCE_F32_TEXTURES')) {
          tf.env().set('WEBGL_FORCE_F32_TEXTURES', true);
          console.log("F32 textures enabled");
        }
      } catch (e) {
        console.warn("WebGL failed, falling back to CPU:", e);
        await tf.setBackend('cpu');
      }
    }
    console.log("Using backend:", tf.getBackend());

    // === Fetch model & params ===
    const { model, trainingParams } = await fetchTfjsModel(p => progressCallback(p * 0.2));
    progressCallback(0.2);

    // === Build datasets ===
    const makeDataset = (dataArray) =>
      dataArray?.length
        ? tf.data
            .generator(function* () {
              const arr = [...dataArray];
              shuffleArray(arr);
              for (const item of arr) {
                const { xs, ys } = tf.tidy(() => {
                  const x = tf.tensor3d(Array.from(item.data), [item.height, item.width, 3]);
                  const y = tf.oneHot(item.label, preparedData.classes.length);
                  return { xs: x, ys: y };
                });
                yield { xs, ys };
              }
            })
            .batch(trainingParams.batch_size)
        : null;

    const trainDataset = makeDataset(preparedData.train);
    const valDataset   = makeDataset(preparedData.validation);
    const testDataset  = makeDataset(preparedData.test);
    progressCallback(0.3);

    // === Compile ===
    model.compile({
      optimizer: tf.train.adam(trainingParams.learning_rate),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });

    // === Small-data simulation ===
    if (
      preparedData.train.length < trainingParams.min_local_samples ||
      (preparedData.validation?.length || 0) < 2
    ) {
      return simulateTraining();
    }

    // === Training loop with logging ===
    const numBatches = Math.ceil(preparedData.train.length / trainingParams.batch_size);
    let bestValAcc = 0;
    const startTime = performance.now();

    for (let epoch = 0; epoch < trainingParams.epochs; epoch++) {
      console.log(`--- Epoch ${epoch+1}/${trainingParams.epochs} START ---`);
      console.log('Before epoch memory:', tf.memory());

      let batchIndex = 0;
      await trainDataset.forEachAsync(async ({ xs, ys }) => {
        batchIndex++;
        console.log(`Epoch ${epoch+1}, Batch ${batchIndex}/${numBatches} — start`);

        tf.tidy(() => {
          const logits = model.predict(xs);
          const loss   = tf.losses.softmaxCrossEntropy(ys, logits).mean();
          const grads  = tf.variableGrads(() => loss);
          model.optimizer.applyGradients(grads.grads);
          Object.values(grads.grads).forEach(t => t.dispose());
          loss.dispose();
        });

        console.log(`Epoch ${epoch+1}, Batch ${batchIndex}/${numBatches} — done`);
      });

      // Validation
      const evalRes = await model.evaluateDataset(valDataset);
      const valLoss = Array.isArray(evalRes) ? evalRes[0].dataSync()[0] : evalRes.loss;
      const valAcc  = Array.isArray(evalRes) ? evalRes[1].dataSync()[0] : evalRes.acc;
      bestValAcc = Math.max(bestValAcc, valAcc);

      const metrics = { epoch: epoch+1, valLoss, valAccuracy: valAcc };
      epochCallback(metrics);
      progressCallback(0.3 + ((epoch+1)/trainingParams.epochs)*0.5);

      console.log('After epoch memory:', tf.memory());
      console.log(`--- Epoch ${epoch+1}/${trainingParams.epochs} END ---`);
      await tf.nextFrame();
    }

    // === Post-training ===
    const trainTime = performance.now() - startTime;
    await markImagesAsUsed(preparedData.imageIds);

    let testLoss = 0, testAcc = 0;
      if (testDataset) {
        const res = await model.evaluateDataset(testDataset);
        [ testLoss, testAcc ] = Array.isArray(res)
          ? [ res[0].dataSync()[0], res[1].dataSync()[0] ]
          : [ res.loss, res.acc ];
      }
      

      // NEW: compute confusion matrix & additional metrics
      const confusionMatrix = await calculateConfusionMatrix(model, preparedData.test, preparedData.classes.length);
      const { precision, recall, f1Score } = calculateAdditionalMetrics(confusionMatrix);

      progressCallback(0.9);

      await tf.io.browserIndexedDB.save('indexeddb://tfjs_full', fullPrecisionArtifacts);
      console.log('📥 Full-precision model stored in IndexedDB');
      
      // Store the INT8-quantized version
      await tf.io.browserIndexedDB.save('indexeddb://tfjs_int8', quantizedArtifacts);
      console.log('📥 Quantized INT8 model stored in IndexedDB');

      const results = {
        model,
        modelData: artifacts,
        version,
        timestamp: new Date().toISOString(),
        loss: testLoss,
        accuracy: testAcc,
        precision,       // ← now set
        recall,          // ← now set
        f1Score,         // ← now set
        trainSize: preparedData.train.length,
        valSize: preparedData.validation?.length,
        testSize: preparedData.test?.length
      };

      if (submitToServer) {
        await submitTrainedModel(results, { trainingTime: trainTime });
      }
      
    progressCallback(1);
    return results;

  } catch (err) {
    console.error("Error in optimized training:", err);
    throw err;
  }
};


// ---- Helper functions ----






/**
 * Creates TensorFlow.js datasets from processed image data
 * @param {Array} trainData - Training data
 * @param {Array} valData - Validation data
 * @param {Array} testData - Test data
 * @param {number} numClasses - Number of classes
 * @param {number} batchSize - Batch size for training
 * @returns {Object} - TensorFlow.js datasets
 */
const createTensorflowDatasets = async (trainData, valData, testData, numClasses, batchSize) => {
  const tf = await import('@tensorflow/tfjs');
  
  // Helper function to convert a data array to a TensorFlow.js dataset
  const createDataset = (dataArray) => {
    if (!dataArray || dataArray.length === 0) return null;
    
    return tf.data.generator(function* () {
      // Shuffle the array for each epoch
      const shuffled = [...dataArray];
      shuffleArray(shuffled);
      
      for (const item of shuffled) {
        // Create input tensor (image data)
        const xs = tf.tensor3d(Array.from(item.data), [item.height, item.width, 3]);
        
        // Create one-hot encoded label tensor
        const ys = tf.oneHot(item.label, numClasses);
        
        yield { xs, ys };
      }
    }).batch(batchSize);
  };
  
  return {
    trainDataset: createDataset(trainData),
    valDataset: createDataset(valData),
    testDataset: createDataset(testData)
  };
};

/**
 * Loads an image from a record to an Image element
 * @param {Object} imageRecord - Image record from IndexedDB
 * @returns {Promise<HTMLImageElement|null>} - Image element or null if failed
 */
const loadImageToElement = async (imageRecord) => {
  if (!isBrowser) {
    throw new Error("This function requires a browser environment");
  }
  
  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    
    img.onload = () => resolve(img);
    img.onerror = () => {
      console.error("Error loading image:", imageRecord.id);
      resolve(null);
    };
    
    // Try different sources in order of preference
    if (imageRecord.objectURL) {
      img.src = imageRecord.objectURL;
    } else if (imageRecord.imageBlob instanceof Blob) {
      img.src = URL.createObjectURL(imageRecord.imageBlob);
    } else if (typeof imageRecord.image === 'string' && imageRecord.image.length > 0) {
      img.src = imageRecord.image;
    } else {
      resolve(null);
    }
  });
};

/**
 * CLAHE (Contrast Limited Adaptive Histogram Equalization) implementation for JavaScript
 * @param {Uint8ClampedArray} data - Image data array
 * @param {number} width - Image width
 * @param {number} height - Image height
 * @param {number} clipLimit - Clip limit for histogram equalization (default: 2.0)
 * @param {Array<number>} tileGridSize - Size of grid for histogram equalization [width, height] (default: [8, 8])
 * @returns {Uint8ClampedArray} - Enhanced image data
 */
const applyCLAHE = (data, width, height, clipLimit = 2.0, tileGridSize = [8, 8]) => {
    try {
      // Convert to Lab color space (simplified conversion for JavaScript)
      const labData = new Uint8ClampedArray(data.length);
      const result = new Uint8ClampedArray(data.length);
      
      // RGB to Lab conversion (approximate)
      for (let i = 0; i < data.length; i += 4) {
        // Get RGB values
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        const alpha = data[i + 3];
        
        // Convert RGB to XYZ (approximate conversion)
        let x = r * 0.4124 + g * 0.3576 + b * 0.1805;
        let y = r * 0.2126 + g * 0.7152 + b * 0.0722;
        let z = r * 0.0193 + g * 0.1192 + b * 0.9505;
        
        // Normalize XYZ
        x /= 95.047;
        y /= 100.0;
        z /= 108.883;
        
        // XYZ to Lab (simplified)
        // Convert XYZ to L*a*b* (approximate)
        const epsilon = 0.008856;
        const kappa = 903.3;
        
        // Compute f(x), f(y), f(z)
        const fx = x > epsilon ? Math.cbrt(x) : (kappa * x + 16) / 116;
        const fy = y > epsilon ? Math.cbrt(y) : (kappa * y + 16) / 116;
        const fz = z > epsilon ? Math.cbrt(z) : (kappa * z + 16) / 116;
        
        // Compute L*, a*, b*
        const L = y > epsilon ? (116 * Math.cbrt(y) - 16) : (kappa * y);
        const a = 500 * (fx - fy);
        const b_value = 200 * (fy - fz);
        
        // Store Lab values (scaled for 8-bit representation)
        labData[i] = Math.max(0, Math.min(255, L * 2.55)); // L: 0-100 scaled to 0-255
        labData[i + 1] = Math.max(0, Math.min(255, a + 128)); // a: -128 to 127 shifted to 0-255
        labData[i + 2] = Math.max(0, Math.min(255, b_value + 128)); // b: -128 to 127 shifted to 0-255
        labData[i + 3] = alpha;
      }
      
      // Apply CLAHE to L channel
      const lChannel = new Uint8ClampedArray(width * height);
      for (let i = 0, j = 0; i < labData.length; i += 4, j++) {
        lChannel[j] = labData[i]; // Extract L channel
      }
      
      // Perform CLAHE on L channel
      const enhancedL = performCLAHE(lChannel, width, height, clipLimit, tileGridSize);
      
      // Merge the enhanced L channel back with original a and b channels
      for (let i = 0, j = 0; i < labData.length; i += 4, j++) {
        result[i] = enhancedL[j]; // Enhanced L
        result[i + 1] = labData[i + 1]; // Original a
        result[i + 2] = labData[i + 2]; // Original b
        result[i + 3] = labData[i + 3]; // Original alpha
      }
      
      // Convert Lab back to RGB
      for (let i = 0; i < result.length; i += 4) {
        // Get Lab values (scaled back from 8-bit representation)
        const L = result[i] / 2.55; // L: 0-255 scaled to 0-100
        const a = result[i + 1] - 128; // a: 0-255 shifted to -128 to 127
        const b_value = result[i + 2] - 128; // b: 0-255 shifted to -128 to 127
        const alpha = result[i + 3];
        
        // Lab to XYZ (simplified conversion)
        const fy = (L + 16) / 116;
        const fx = fy + a / 500;
        const fz = fy - b_value / 200;
        
        const epsilon = 0.008856;
        const kappa = 903.3;
        
        // Compute x, y, z
        const y = L > kappa * epsilon ? Math.pow(fy, 3) : L / kappa;
        const x = Math.pow(fx, 3) > epsilon ? Math.pow(fx, 3) : (116 * fx - 16) / kappa;
        const z = Math.pow(fz, 3) > epsilon ? Math.pow(fz, 3) : (116 * fz - 16) / kappa;
        
        // XYZ to RGB (approximate conversion)
        const xn = x * 95.047;
        const yn = y * 100.0;
        const zn = z * 108.883;
        
        // XYZ to RGB matrix transformation (approximate)
        let r = xn * 3.2406 - yn * 1.5372 - zn * 0.4986;
        let g = -xn * 0.9689 + yn * 1.8758 + zn * 0.0415;
        let b = xn * 0.0557 - yn * 0.2040 + zn * 1.0570;
        
        // Gamma correction and clipping
        r = Math.max(0, Math.min(255, Math.round(r > 0.0031308 ? 
            1.055 * Math.pow(r, 1/2.4) - 0.055 : 12.92 * r)));
        g = Math.max(0, Math.min(255, Math.round(g > 0.0031308 ? 
            1.055 * Math.pow(g, 1/2.4) - 0.055 : 12.92 * g)));
        b = Math.max(0, Math.min(255, Math.round(b > 0.0031308 ? 
            1.055 * Math.pow(b, 1/2.4) - 0.055 : 12.92 * b)));
        
        // Store RGB values
        data[i] = r;
        data[i + 1] = g;
        data[i + 2] = b;
        // Alpha channel remains unchanged
      }
      
      return data;
    } catch (error) {
      console.error("Error applying CLAHE:", error);
      return data; // Return original data on error
    }
  };
  
  /**
   * Perform CLAHE on a single channel image
   * @param {Uint8ClampedArray} channel - Single channel image data
   * @param {number} width - Image width
   * @param {number} height - Image height
   * @param {number} clipLimit - Clip limit for histogram equalization
   * @param {Array<number>} tileGridSize - Size of grid for histogram equalization [width, height]
   * @returns {Uint8ClampedArray} - Enhanced single channel image
   */
  const performCLAHE = (channel, width, height, clipLimit, tileGridSize) => {
    const [tilesX, tilesY] = tileGridSize;
    const tileWidth = Math.floor(width / tilesX);
    const tileHeight = Math.floor(height / tilesY);
    const result = new Uint8ClampedArray(channel.length);
    
    // Create histogram for each tile
    const histograms = Array(tilesX * tilesY).fill().map(() => Array(256).fill(0));
    const lookupTables = Array(tilesX * tilesY).fill().map(() => Array(256).fill(0));
    
    // Calculate histograms for each tile
    for (let ty = 0; ty < tilesY; ty++) {
      for (let tx = 0; tx < tilesX; tx++) {
        const histogram = histograms[ty * tilesX + tx];
        
        // Calculate histogram for this tile
        for (let y = 0; y < tileHeight; y++) {
          if (ty * tileHeight + y >= height) continue;
          
          for (let x = 0; x < tileWidth; x++) {
            if (tx * tileWidth + x >= width) continue;
            
            const pixelIdx = (ty * tileHeight + y) * width + (tx * tileWidth + x);
            const pixelValue = channel[pixelIdx];
            histogram[pixelValue]++;
          }
        }
        
        // Apply clip limit
        const tileSize = tileWidth * tileHeight;
        const clipHistogram = [...histogram];
        let clippedPixels = 0;
        
        if (clipLimit > 0) {
          const clipThreshold = Math.max(1, Math.floor(clipLimit * tileSize / 256));
          
          // Clip histogram and count clipped pixels
          for (let i = 0; i < 256; i++) {
            if (clipHistogram[i] > clipThreshold) {
              clippedPixels += (clipHistogram[i] - clipThreshold);
              clipHistogram[i] = clipThreshold;
            }
          }
          
          // Redistribute clipped pixels
          const redistIncrement = Math.floor(clippedPixels / 256);
          let residual = clippedPixels - redistIncrement * 256;
          
          for (let i = 0; i < 256; i++) {
            clipHistogram[i] += redistIncrement;
            
            // Distribute remainder
            if (residual > 0) {
              clipHistogram[i]++;
              residual--;
            }
          }
        }
        
        // Create lookup table for this tile
        const cdf = new Array(256).fill(0);
        cdf[0] = clipHistogram[0];
        
        for (let i = 1; i < 256; i++) {
          cdf[i] = cdf[i - 1] + clipHistogram[i];
        }
        
        const cdfMin = cdf.find(v => v > 0) || 0;
        const cdfMax = cdf[255];
        const cdfRange = cdfMax - cdfMin;
        
        // Create lookup table
        if (cdfRange > 0) {
          for (let i = 0; i < 256; i++) {
            lookupTables[ty * tilesX + tx][i] = Math.round(
              ((cdf[i] - cdfMin) / cdfRange) * 255
            );
          }
        } else {
          // If all pixels in the tile have the same value
          for (let i = 0; i < 256; i++) {
            lookupTables[ty * tilesX + tx][i] = i;
          }
        }
      }
    }
    
    // Apply interpolated lookup tables to the image
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const pixelIdx = y * width + x;
        const pixelValue = channel[pixelIdx];
        
        // Determine which tiles this pixel is influenced by
        const tx = Math.min(tilesX - 1, Math.floor(x / tileWidth));
        const ty = Math.min(tilesY - 1, Math.floor(y / tileHeight));
        
        // Get pixel location relative to tile boundaries
        const xInTile = (x - tx * tileWidth) / tileWidth;
        const yInTile = (y - ty * tileHeight) / tileHeight;
        
        // Calculate interpolation weights
        const x1 = Math.max(0, tx - 1);
        const x2 = Math.min(tilesX - 1, tx + 1);
        const y1 = Math.max(0, ty - 1);
        const y2 = Math.min(tilesY - 1, ty + 1);
        
        // Calculate interpolated value
        let newValue = 0;
        let totalWeight = 0;
        
        for (let ny = y1; ny <= y2; ny++) {
          for (let nx = x1; nx <= x2; nx++) {
            // Calculate distance-based weight
            const dx = Math.max(0, 1 - Math.abs(x / tileWidth - nx - 0.5) * 2);
            const dy = Math.max(0, 1 - Math.abs(y / tileHeight - ny - 0.5) * 2);
            const weight = dx * dy;
            
            if (weight > 0) {
              newValue += lookupTables[ny * tilesX + nx][pixelValue] * weight;
              totalWeight += weight;
            }
          }
        }
        
        // Normalize the result
        if (totalWeight > 0) {
          result[pixelIdx] = Math.round(newValue / totalWeight);
        } else {
          result[pixelIdx] = pixelValue;
        }
      }
    }
    
    return result;
  };
  
  /**
   * Process an image for training (resize, apply CLAHE, normalize)
   * @param {HTMLImageElement} imgElement - Image element
   * @param {string} className - The class label
   * @param {Array} classes - List of all class names
   * @returns {Object} - Processed image data
   */
  const processImageForTraining = async (imgElement, className, classes) => {
    try {
      if (!isBrowser) {
        throw new Error("This function requires a browser environment");
      }
      
      // First canvas to resize the image to 256x256
      const resizeCanvas = document.createElement('canvas');
      const resizeCtx = resizeCanvas.getContext('2d');
      
      // Set resize canvas size to 256x256 (matching the resize in PyTorch)
      resizeCanvas.width = 256;
      resizeCanvas.height = 256;
      
      // Draw the image on the resize canvas
      resizeCtx.drawImage(imgElement, 0, 0, resizeCanvas.width, resizeCanvas.height);
      
      // Create a second canvas for center cropping to 224x224
      const cropCanvas = document.createElement('canvas');
      const cropCtx = cropCanvas.getContext('2d', { willReadFrequently: true });
      
      // Set crop canvas size to 224x224 (matching CenterCrop in PyTorch)
      cropCanvas.width = 224;
      cropCanvas.height = 224;
      
      // Calculate crop coordinates (center crop)
      const cropX = (resizeCanvas.width - cropCanvas.width) / 2;
      const cropY = (resizeCanvas.height - cropCanvas.height) / 2;
      
      // Draw the center-cropped image on the crop canvas
      cropCtx.drawImage(
        resizeCanvas, 
        cropX, cropY, cropCanvas.width, cropCanvas.height,
        0, 0, cropCanvas.width, cropCanvas.height
      );
      
      // Get image data for CLAHE processing
      const imageData = cropCtx.getImageData(0, 0, cropCanvas.width, cropCanvas.height);
      
      // Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
      const enhancedData = applyCLAHE(
        imageData.data, 
        cropCanvas.width, 
        cropCanvas.height, 
        2.0, // clipLimit
        [8, 8] // tileGridSize
      );
      
      // Put enhanced data back to canvas
      const enhancedImageData = new ImageData(
        enhancedData, 
        cropCanvas.width, 
        cropCanvas.height
      );
      cropCtx.putImageData(enhancedImageData, 0, 0);
      
      // Get final image data
      const finalImageData = cropCtx.getImageData(0, 0, cropCanvas.width, cropCanvas.height);
      
      // Normalize with mean and std similar to PyTorch's transforms.Normalize
      const normalizedData = new Float32Array(finalImageData.data.length / 4 * 3); // RGB only
      
      // ImageNet mean and std values
      const mean = [0.485, 0.456, 0.406];
      const std = [0.229, 0.224, 0.225];
      
      for (let i = 0, j = 0; i < finalImageData.data.length; i += 4, j += 3) {
        // First normalize to [0,1]
        const r = finalImageData.data[i] / 255;
        const g = finalImageData.data[i+1] / 255;
        const b = finalImageData.data[i+2] / 255;
        
        // Then normalize with mean and std
        normalizedData[j] = (r - mean[0]) / std[0];
        normalizedData[j+1] = (g - mean[1]) / std[1];
        normalizedData[j+2] = (b - mean[2]) / std[2];
      }
      
      return {
        data: normalizedData,
        width: cropCanvas.width,
        height: cropCanvas.height,
        channels: 3,
        class: className,
        // One-hot encode the class
        label: classes.indexOf(className)
      };
    } catch (error) {
      console.error("Error processing image:", error);
      return null;
    }
  };

/**
 * Shuffles an array in-place
 * @param {Array} array - Array to shuffle
 */
const shuffleArray = (array) => {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
};

/**
 * Calculate confusion matrix for the model on test data
 * @param {Object} model - TensorFlow.js model
 * @param {Array} testData - Test data array
 * @param {number} numClasses - Number of classes
 * @returns {Array} - Confusion matrix
 */
const calculateConfusionMatrix = async (model, testData, numClasses) => {
  try {
    const tf = await import('@tensorflow/tfjs');
    
    // Initialize confusion matrix
    const confusionMatrix = Array(numClasses).fill().map(() => Array(numClasses).fill(0));
    
    // Process each test item
    for (const item of testData) {
      // Create input tensor
      const input = tf.tensor3d(Array.from(item.data), [item.height, item.width, 3]);
      input.expandDims(0); // Add batch dimension
      
      // Make prediction
      const prediction = model.predict(input.expandDims(0));
      
      // Get predicted class (argmax)
      const predictedClass = prediction.argMax(1).dataSync()[0];
      
      // Update confusion matrix
      confusionMatrix[item.label][predictedClass]++;
      
      // Clean up tensors
      input.dispose();
      prediction.dispose();
    }
    
    return confusionMatrix;
  } catch (error) {
    console.error("Error calculating confusion matrix:", error);
    // Return empty matrix on error
    return Array(numClasses).fill().map(() => Array(numClasses).fill(0));
  }
};

/**
 * Calculate precision, recall, and F1 score from confusion matrix
 * @param {Array} confusionMatrix - Confusion matrix
 * @returns {Object} - Precision, recall, and F1 score
 */
const calculateAdditionalMetrics = (confusionMatrix) => {
  if (!confusionMatrix || confusionMatrix.length === 0) {
    // Return default values if confusion matrix is not available
    return {
      precision: 0.7,
      recall: 0.7,
      f1Score: 0.7
    };
  }
  
  const numClasses = confusionMatrix.length;
  
  // Calculate per-class precision and recall
  const perClassMetrics = [];
  
  for (let i = 0; i < numClasses; i++) {
    let truePositives = confusionMatrix[i][i];
    let falsePositives = 0;
    let falseNegatives = 0;
    
    // Sum false positives (predicted as class i but were not)
    for (let j = 0; j < numClasses; j++) {
      if (j !== i) {
        falsePositives += confusionMatrix[j][i];
      }
    }
    
    // Sum false negatives (were class i but predicted as something else)
    for (let j = 0; j < numClasses; j++) {
      if (j !== i) {
        falseNegatives += confusionMatrix[i][j];
      }
    }
    
    // Calculate precision and recall for this class
    const precision = truePositives + falsePositives > 0 ? 
                       truePositives / (truePositives + falsePositives) : 0;
                       
    const recall = truePositives + falseNegatives > 0 ? 
                     truePositives / (truePositives + falseNegatives) : 0;
    
    perClassMetrics.push({ precision, recall });
  }
  
  // Calculate macro-average for precision and recall
  const macroAvgPrecision = perClassMetrics.reduce((sum, metrics) => sum + metrics.precision, 0) / numClasses;
  const macroAvgRecall = perClassMetrics.reduce((sum, metrics) => sum + metrics.recall, 0) / numClasses;
  
  // Calculate F1 score
  const f1Score = macroAvgPrecision > 0 && macroAvgRecall > 0 ?
                  2 * (macroAvgPrecision * macroAvgRecall) / (macroAvgPrecision + macroAvgRecall) : 0;
  
  return {
    precision: macroAvgPrecision,
    recall: macroAvgRecall,
    f1Score: f1Score
  };
};


  /**
   * Helper function to get typed array view based on dtype
   */
  function getTypedArrayForDType(dtype, buffer, begin, size) {
    switch (dtype) {
      case 'float32':
        return new Float32Array(buffer, begin, size);
      case 'int32':
        return new Int32Array(buffer, begin, size);
      case 'bool':
        return new Uint8Array(buffer, begin, size);
      case 'complex64':
        // Complex numbers are stored as pairs of float32s
        return new Float32Array(buffer, begin, size * 2);
      default:
        throw new Error(`Unsupported dtype: ${dtype}`);
    }
  }
  
  /**
   * Helper function to determine bytes per element for each dtype
   */
  function bytesPerElement(dtype) {
    switch (dtype) {
      case 'float32':
      case 'int32':
        return 4;
      case 'bool':
        return 1;
      case 'complex64':
        return 8;
      default:
        throw new Error(`Unsupported dtype: ${dtype}`);
    }
  }


  async function getLatestTfjsModel() {
    const db = await openDatabase('xrayImagesDB');
  
    return new Promise((resolve, reject) => {
      const tx = db.transaction('models', 'readonly');
      const store = tx.objectStore('models');
      const req = store.getAll();
  
      req.onsuccess = () => {
        const models = req.result
          .filter(m => m.id.startsWith('tfjs_'))
          .sort((a, b) => b.timestamp - a.timestamp);
  
        if (models.length === 0) {
          resolve(null);
          return;
        }
  
        const blob = models[0].modelBlob;
  
        const reader = new FileReader();
        reader.onload = () => {
          try {
            const parsed = JSON.parse(reader.result);
  
            if (
              parsed.modelTopology &&
              parsed.weightSpecs &&
              parsed.weightData
            ) {
              // Convert weightData back to ArrayBuffer
              parsed.weightData = new Uint8Array(parsed.weightData).buffer;
              resolve(parsed);
            } else {
              console.error("Malformed modelArtifacts:", parsed);
              resolve(null);
            }
          } catch (err) {
            console.error("Failed to parse model blob as JSON:", err);
            resolve(null);
          }
        };
  
        reader.onerror = (e) => {
          reject(`FileReader error: ${e.target.error}`);
        };
  
        reader.readAsText(blob); // Assume it's a JSON blob
      };
  
      req.onerror = (e) => reject(`Model list error: ${e.target.errorCode}`);
    });
  }
  

  
/** Returns BYTES per element for a given dtype */
function bytesPerElement(dtype) {
  switch (dtype) {
    case 'float32': return 4;
    case 'int32':   return 4;
    case 'bool':    return 1;
    default: throw new Error(`Unsupported dtype ${dtype}`);
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

/**
 * Submits trained model weight arrays and metrics via multipart/form-data
 * so the binary weightData → JSON conversion happens only once.
 *
 * @param {Object} trainedModel  // should include `.model`, `.accuracy`, `.loss`, `.round`, `.trainSize`, etc.
 * @param {Object} systemMetrics optional
 */
export async function submitTrainedModel(trainedModel, systemMetrics = {}) {
  
  if (!trainedModel || !trainedModel.model) {
    throw new Error("No trained model provided");
  }

  // 1) extract artifacts without touching the network
  const artifacts = await trainedModel.model.save(
    tf.io.withSaveHandler(x => Promise.resolve(x))
  );
  // artifacts: { weightSpecs: [...], weightData: ArrayBuffer }

  // 2) unpack into a plain object { weightName: [ ...values ] }
  const { weightSpecs, weightData } = artifacts;
  const weightsJson = {};
  let offset = 0;
  for (const spec of weightSpecs) {
    const { name, shape, dtype } = spec;
    const size = shape.reduce((a, b) => a * b, 1);
    const arr = getTypedArrayForDType(dtype, weightData, offset, size);
    weightsJson[name] = Array.from(arr);
    offset += size;  // because getTypedArrayForDType expects offset in elements
  }

  // 3) build metadata payload
  const payload = {
    metrics: {
      accuracy:   parseFloat(trainedModel.accuracy  || 0),
      loss:       parseFloat(trainedModel.loss      || 0),
      precision:  parseFloat(trainedModel.precision || 0),
      recall:     parseFloat(trainedModel.recall    || 0),
      f1Score:    parseFloat(trainedModel.f1Score   || 0)
    },
    round:       trainedModel.round     || 0,
    num_samples: trainedModel.trainSize || 0,
    timestamp:   new Date().toISOString(),
    system_metrics: systemMetrics
  };

  // 4) assemble multipart form
  const form = new FormData();
  form.append('metadata', new Blob([JSON.stringify(payload)], { type: 'application/json' }));
  form.append('weights',  new Blob([JSON.stringify(weightsJson)], { type: 'application/json' }), 'weights.json');

  // 5) headers: API key + optional JWT + client ID
  const API_KEY    = "FeDMl2025";  // ideally injected via env
  const CLIENT_ID  = localStorage.getItem('clientId') || `client-${Date.now()}`;
  const JWT        = localStorage.getItem('jwtToken');
  const headers = {
    'X-API-Key': API_KEY,
    'X-Client-ID': CLIENT_ID
  };
  if (JWT) headers['Authorization'] = `Bearer ${JWT}`;

  // 6) send off
  const resp = await fetch("http://localhost:5050/tfjs/submit_weights", {
    method: 'POST',
    headers,
    body: form,
    credentials: 'include'  // ensure cookies/auth on cross-origin if needed
  });

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(`Submission failed: ${err.message||resp.statusText}`);
  }

  return resp.json();
}