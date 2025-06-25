import * as tf from '@tensorflow/tfjs';
import { processImagesInBatches } from '../utils/indexedDBUtils.js';

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
export const ds = tf.data
    .generator(() => indexedDBGenerator(64))
    .shuffle(32 * 2)        // buffer twice your batch
    .batch(32)
    .prefetch( 4 );

/**
 * Shuffles an array in-place
 * @param {Array} array - Array to shuffle
 */
export const shuffleArray = (array) => {
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
export const calculateConfusionMatrix = async (model, testData, numClasses) => {
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