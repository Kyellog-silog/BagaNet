//  Batch-oriented IndexedDB utilities with Blob/File support

/**
 * Convert a base64 string to a Blob object
 * @param {string} base64String - The base64 string (with or without data:image prefix)
 * @returns {Blob} - The Blob representation of the data
 */
export const base64ToBlob = (base64String) => {
  // Handle both plain base64 strings and data URLs
  const base64Data = base64String.startsWith('data:') 
    ? base64String.split(',')[1] 
    : base64String;
  
  // Decode base64
  const binaryString = atob(base64Data);
  const bytes = new Uint8Array(binaryString.length);
  
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  
  // Detect content type from data URL or default to jpeg
  let contentType = 'image/jpeg';
  if (base64String.startsWith('data:')) {
    contentType = base64String.split(',')[0].split(':')[1].split(';')[0];
  }
  
  return new Blob([bytes], { type: contentType });
};

/**
 * Checks if a specific object store exists in the database
 * @param {string} storeName - Name of the store to check
 * @param {string} dbName - Database name
 * @returns {Promise<boolean>} - True if store exists, false otherwise
 */
export const storeExists = async (storeName, dbName = 'xrayImagesDB') => {
  return new Promise((resolve) => {
    const request = indexedDB.open(dbName);
    
    request.onsuccess = (e) => {
      const db = e.target.result;
      const exists = db.objectStoreNames.contains(storeName);
      db.close();
      resolve(exists);
    };
    
    request.onerror = () => {
      resolve(false); // Assume it doesn't exist if we can't open the DB
    };
  });
};

/**
 * Gets the current version of the database
 * @param {string} dbName - Database name
 * @returns {Promise<number>} - Current version number
 */
export const getDatabaseVersion = async (dbName = 'xrayImagesDB') => {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(dbName);
    
    request.onsuccess = (e) => {
      const version = e.target.result.version;
      e.target.result.close();
      resolve(version);
    };
    
    request.onerror = (e) => {
      reject(`Error getting database version: ${e.target.errorCode}`);
    };
  });
};

/**
 * Opens (or upgrades) the IndexedDB database
 */
export const openDatabase = async (dbName = 'xrayImagesDB', version = null) => {
  // If no version specified, try to determine if we need to upgrade
  if (version === null) {
    try {
      // Check if database exists
      const existingVersion = await getDatabaseVersion(dbName);
      const hasModelsStore = await storeExists('models', dbName);
      
      // If models store doesn't exist, increment version to trigger upgrade
      if (!hasModelsStore) {
        version = existingVersion + 1;
        console.log(`Upgrading database from version ${existingVersion} to ${version} to add 'models' store`);
      } else {
        version = existingVersion;
      }
    } catch (err) {
      // Default to version 1 if error
      version = 1;
    }
  }

  return new Promise((resolve, reject) => {
    const request = indexedDB.open(dbName, version);

    request.onerror = (e) => reject(`DB error: ${e.target.errorCode}`);
    request.onsuccess = (e) => resolve(e.target.result);

    // On upgrade: create store and indexes
    request.onupgradeneeded = (e) => {
      const db = e.target.result;
      if (!db.objectStoreNames.contains('images')) {
        const store = db.createObjectStore('images', { keyPath: 'id' });
        // Create essential indexes for querying
        store.createIndex('patientId', 'patientId', { unique: false });
        store.createIndex('timestamp', 'timestamp', { unique: false });
        store.createIndex('isEvaluated', 'isEvaluated', { unique: false });
        store.createIndex('classification', 'classification', { unique: false });
        store.createIndex('trainingStatus', 'trainingStatus', { unique: false });
      } else if (e.oldVersion < version) {
        // Handle upgrade for existing stores
        const transaction = e.target.transaction;
        const store = transaction.objectStore('images');
        
        // Add trainingStatus index if it doesn't exist
        if (!store.indexNames.contains('trainingStatus')) {
          store.createIndex('trainingStatus', 'trainingStatus', { unique: false });
        }
        
        // Add classification index if it doesn't exist
        if (!store.indexNames.contains('classification')) {
          store.createIndex('classification', 'classification', { unique: false });
        }
      }
      
      // Always ensure models store exists during any upgrade
      if (!db.objectStoreNames.contains('models')) {
        console.log("Creating 'models' object store");
        db.createObjectStore('models', { keyPath: 'id' });
      }
    };
  });
};

/**
 * Save a model Blob (or ArrayBuffer) under a fixed id (e.g. 'default').
 * @param {Blob|ArrayBuffer} modelData
 * @param {string} id
 */
export const storeModel = async (modelData, id = 'default', dbName = 'xrayImagesDB') => {
  // First ensure the models store exists
  try {
    const hasModelsStore = await storeExists('models', dbName);
    if (!hasModelsStore) {
      console.log("'models' store doesn't exist, upgrading database");
      const currentVersion = await getDatabaseVersion(dbName);
      await openDatabase(dbName, currentVersion + 1);
    }
  } catch (err) {
    console.error("Error checking for models store:", err);
  }

  const blob = modelData instanceof Blob ? modelData : new Blob([modelData], { type: 'application/octet-stream' });
  const db = await openDatabase(dbName);
  return new Promise((resolve, reject) => {
    const tx = db.transaction(['models'], 'readwrite');
    const store = tx.objectStore('models');
    const record = { id, modelBlob: blob, timestamp: Date.now() };
    const req = store.put(record);
    req.onsuccess = () => resolve(record);
    req.onerror = (e) => reject(`Model store error: ${e.target.errorCode}`);
    tx.oncomplete = () => db.close();
  });
};

/**
 * Retrieve a model Blob by id. Resolves to the Blob or null if not found.
 * @param {string} id
 */
export const getModel = async (id = 'default', dbName = 'xrayImagesDB') => {
  try {
    // First check if the models store exists
    const hasModelsStore = await storeExists('models', dbName);
    if (!hasModelsStore) {
      console.log("'models' store doesn't exist, will return null");
      return null;
    }
    
    const db = await openDatabase(dbName);
    return new Promise((resolve, reject) => {
      const tx = db.transaction('models', 'readonly');
      const store = tx.objectStore('models');
      const req = store.get(id);
      req.onsuccess = () => {
        const result = req.result ? req.result.modelBlob : null;
        resolve(result);
      };
      req.onerror = (e) => reject(`Model get error: ${e.target.errorCode}`);
      tx.oncomplete = () => db.close();
    });
  } catch (err) {
    console.error("Error in getModel:", err);
    return null; // Return null if any error occurs
  }
};


/**
 * Store a File/Blob or Base64 string with optional compression and metadata
 * @param {Blob|string} input - File/Blob or dataURL string
 * @param {Object} metadata - e.g. { patientId, evaluation, classification, ... }
 * @param {boolean} compress - whether to resize/compress (only for dataURL)
 */
export const storeImage = async (
  input,
  metadata = {},
  compress = false,
  dbName = 'xrayImagesDB'
) => {
  let blob;

  // 1) Normalize to Blob
  if (input instanceof Blob) {
    blob = input;
  } else if (typeof input === 'string' && input.startsWith('data:image/')) {
    // dataURL -> Blob
    const res = await fetch(input);
    blob = await res.blob();
  } else if (typeof input === 'string') {
    // Assume base64 string
    blob = base64ToBlob(input);
  } else {
    throw new Error('Unsupported image input type');
  }

  // 2) Optional compression via canvas
  if (compress) {
    blob = await compressBlob(blob);
  }

  // 3) Prepare record with minimal required fields
  const id = metadata.id || Date.now().toString();
  const timestamp = metadata.timestamp || Date.now();
  
  const record = {
    id,
    imageBlob: blob,
    // Don't store objectURL in IndexedDB, it will be generated on demand
    timestamp,
    
    // Essential fields with defaults
    isEvaluated: metadata.isEvaluated || false,
    trainingStatus: metadata.trainingStatus || 'unused',
    classification: metadata.classification || 'Unclassified',
    
    // Include any other metadata provided
    ...metadata
  };

  // 4) Store in IndexedDB
  const db = await openDatabase(dbName);
  return new Promise((resolve, reject) => {
    const tx = db.transaction(['images'], 'readwrite');
    const store = tx.objectStore('images');
    const req = store.put(record);
    
    req.onsuccess = () => resolve(id);
    req.onerror = (e) => reject(`Store error: ${e.target.errorCode}`);
    
    // Ensure transaction completes
    tx.oncomplete = () => console.log(`Transaction completed, image ${id} saved`);
    tx.onerror = (e) => console.error('Transaction error:', e);
  });
};


/**
 * Compress a Blob via canvas (max 600px, quality 0.7)
 */
const compressBlob = (blob) => {
  return new Promise((resolve) => {
    const img = new Image();
    const url = URL.createObjectURL(blob);

    img.onload = () => {
      let { width, height } = img;
      const maxDim = 600;
      if (width > height && width > maxDim) {
        height = (height * maxDim) / width;
        width = maxDim;
      } else if (height > maxDim) {
        width = (width * maxDim) / height;
        height = maxDim;
      }
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, width, height);

      canvas.toBlob(
        (compressed) => {
          URL.revokeObjectURL(url);
          resolve(compressed);
        },
        'image/jpeg',
        0.7
      );
    };

    img.src = url;
  });
};

/**
 * Create object URL from blob and add it to the image object
 * @param {Object} image - Image object from IndexedDB
 * @returns {Object} - Image with added objectURL property
 */
export const addObjectURL = (image) => {
  // Skip if no imageBlob or if objectURL already exists and is valid
  if (!image || !image.imageBlob) return image;
  
  // Revoke previous objectURL if it exists to prevent memory leaks
  if (image.objectURL) {
    try {
      URL.revokeObjectURL(image.objectURL);
    } catch (e) {
      console.warn('Failed to revoke objectURL:', e);
    }
  }
  
  // Create a fresh objectURL from the blob
  image.objectURL = URL.createObjectURL(image.imageBlob);
  return image;
};

/**
 * Process object URLs for an array of images
 * @param {Array} images - Array of image objects from IndexedDB
 * @returns {Array} - Images with added objectURL properties
 */
export const addObjectURLs = (images) => {
  if (!images || !Array.isArray(images)) return [];
  return images.map(addObjectURL);
};



/**
 * Fetch all keys (IDs)
 */
export const getAllImageIds = async (dbName = 'xrayImagesDB') => {
  const db = await openDatabase(dbName);
  return new Promise((resolve, reject) => {
    const tx = db.transaction('images', 'readonly');
    const store = tx.objectStore('images');
    const req = store.getAllKeys();
    req.onsuccess = () => resolve(req.result);
    req.onerror = (e) => reject(e.target.errorCode);
  });
};


/**
 * Get a batch of records by index
 */
export const getImagesBatch = async (
  dbName = 'xrayImagesDB',
  batchSize = 10,
  startIndex = 0
) => {
  const allIds = await getAllImageIds(dbName);
  const batchIds = allIds.slice(startIndex, startIndex + batchSize);
  if (!batchIds.length) return [];

  const db = await openDatabase(dbName);
  const images = [];
  for (const id of batchIds) {
    const rec = await new Promise((res, rej) => {
      const tx = db.transaction('images', 'readonly');
      const store = tx.objectStore('images');
      const req = store.get(id);
      req.onsuccess = () => {
        // Add fresh objectURL before returning
        const image = addObjectURL(req.result);
        res(image);
      };
      req.onerror = (e) => rej(e.target.errorCode);
    });
    images.push(rec);
  }
  return images;
};



/**
 * Process all images in batches via a callback
 */
export const processImagesInBatches = async (
  processFn,
  batchSize = 10,
  dbName = 'xrayImagesDB'
) => {
  const ids = await getAllImageIds(dbName);
  const results = [];
  for (let i = 0; i < ids.length; i += batchSize) {
    const batch = await getImagesBatch(dbName, batchSize, i);
    const batchRes = await processFn(batch);
    results.push(...batchRes);
  }
  return results;
};

/**
 * Shortcut: retrieve everything via batches (default size 20)
 */
export const getAllStoredImages = async (dbName = 'xrayImagesDB') => {
  return processImagesInBatches(batch => batch, 20, dbName);
};

/**
 * Retrieve images by patient ID
 * @param {string} patientId - The patient ID to filter by
 * @param {string} dbName - Database name
 */
export const getImagesByPatientId = async (patientId, dbName = 'xrayImagesDB') => {
  const db = await openDatabase(dbName);
  return new Promise((resolve, reject) => {
    const tx = db.transaction('images', 'readonly');
    const store = tx.objectStore('images');
    const index = store.index('patientId');
    const req = index.getAll(patientId);
    
    req.onsuccess = () => {
      // Add fresh objectURLs to all images
      const images = addObjectURLs(req.result);
      resolve(images);
    };
    req.onerror = (e) => reject(e.target.errorCode);
  });
};

/**
 * Retrieve only evaluated images
 * @param {string} dbName - Database name
 */
export const getEvaluatedImages = async (dbName = 'xrayImagesDB') => {
  const db = await openDatabase(dbName);
  return new Promise((resolve, reject) => {
    const tx = db.transaction('images', 'readonly');
    const store = tx.objectStore('images');
    const index = store.index('isEvaluated');
    const req = index.getAll(true);
    
    req.onsuccess = () => {
      // Add fresh objectURLs to all images
      const images = addObjectURLs(req.result);
      resolve(images);
    };
    req.onerror = (e) => reject(e.target.errorCode);
  });
};

/**
 * Retrieve evaluated images by patient ID
 * @param {string} patientId - The patient ID to filter by
 * @param {string} dbName - Database name
 */
export const getEvaluatedImagesByPatientId = async (patientId, dbName = 'xrayImagesDB') => {
  const allPatientImages = await getImagesByPatientId(patientId, dbName);
  return allPatientImages.filter(image => image.isEvaluated === true);
};

/**
 * Retrieve images by classification
 * @param {string} classification - The classification to filter by (e.g., 'Normal', 'Pneumonia')
 * @param {string} dbName - Database name
 */
export const getImagesByClassification = async (classification, dbName = 'xrayImagesDB') => {
  const db = await openDatabase(dbName);
  return new Promise((resolve, reject) => {
    const tx = db.transaction('images', 'readonly');
    const store = tx.objectStore('images');
    const index = store.index('classification');
    const req = index.getAll(classification);
    
    req.onsuccess = () => {
      // Add fresh objectURLs to all images
      const images = addObjectURLs(req.result);
      resolve(images);
    };
    req.onerror = (e) => reject(e.target.errorCode);
  });
};

/**
 * Retrieve images by training status
 * @param {string} status - The training status to filter by ('unused', 'used', 'purged')
 * @param {string} dbName - Database name
 */
export const getImagesByTrainingStatus = async (status, dbName = 'xrayImagesDB') => {
  const db = await openDatabase(dbName);
  return new Promise((resolve, reject) => {
    const tx = db.transaction('images', 'readonly');
    const store = tx.objectStore('images');
    
    // Check if the trainingStatus index exists
    if (!store.indexNames.contains('trainingStatus')) {
      // Return empty array if index doesn't exist yet
      resolve([]);
      return;
    }
    
    const index = store.index('trainingStatus');
    const req = index.getAll(status);
    
    req.onsuccess = () => {
      // Add fresh objectURLs to all images
      const images = addObjectURLs(req.result);
      resolve(images);
    };
    req.onerror = (e) => reject(e.target.errorCode);
  });
};

/**
 * Get unused images that haven't been used for training
 * @param {string} dbName - Database name
 */
export const getUnusedImages = async (dbName = 'xrayImagesDB') => {
  return getImagesByTrainingStatus('unused', dbName);
};

/**
 * Mark images as used after training
 * @param {Array} imageIds - Array of image IDs to mark as used
 * @param {string} dbName - Database name
 */
export const markImagesAsUsed = async (imageIds, dbName = 'xrayImagesDB') => {
  if (!imageIds || imageIds.length === 0) return;
  
  const db = await openDatabase(dbName);
  const tx = db.transaction(['images'], 'readwrite');
  const store = tx.objectStore('images');
  
  const timestamp = new Date();
  const promises = imageIds.map(id => {
    return new Promise((resolve, reject) => {
      // Get current record
      const getReq = store.get(id);
      
      getReq.onsuccess = (e) => {
        const record = e.target.result;
        if (record) {
          // Update training status and timestamp
          record.trainingStatus = 'used';
          record.lastTrainingDate = timestamp;
          
          // Put updated record back
          const putReq = store.put(record);
          putReq.onsuccess = () => resolve(id);
          putReq.onerror = (err) => reject(err);
        } else {
          resolve(null); // Image not found
        }
      };
      
      getReq.onerror = (err) => reject(err);
    });
  });
  
  return Promise.all(promises);
};

/**
 * Purge images that have been used for training
 * @param {string} dbName - Database name
 * @returns {Promise<number>} - Number of purged images
 */
export const purgeUsedImages = async (dbName = 'xrayImagesDB') => {
  const usedImages = await getImagesByTrainingStatus('used', dbName);
  if (!usedImages || usedImages.length === 0) return 0;
  
  const db = await openDatabase(dbName);
  const tx = db.transaction(['images'], 'readwrite');
  const store = tx.objectStore('images');
  
  const promises = usedImages.map(image => {
    return new Promise((resolve, reject) => {
      // Mark as 'purged' for record-keeping
      const record = { ...image };
      // Remove the objectURL before storing (it will be regenerated when needed)
      delete record.objectURL;
      record.trainingStatus = 'purged';
      
      const putReq = store.put(record);
      putReq.onsuccess = () => resolve(image.id);
      putReq.onerror = (err) => reject(err);
    });
  });
  
  await Promise.all(promises);
  
  // Revoke all object URLs from the in-memory images
  revokeObjectURLs(usedImages);
  
  return usedImages.length;
};

/**
 * Utility to revoke all objectURLs to prevent memory leaks
 * @param {Array} images - Array of image objects with objectURL properties
 */
export const revokeObjectURLs = (images) => {
  if (!images || !Array.isArray(images)) return;
  
  images.forEach(image => {
    if (image && image.objectURL) {
      try {
        URL.revokeObjectURL(image.objectURL);
        // Delete the objectURL property since it's now invalid
        delete image.objectURL;
      } catch (e) {
        console.warn('Failed to revoke objectURL:', e);
      }
    }
  });
};

/**
 * Save an image to IndexedDB
 * Helper alias for more readable code
 */
export const saveImageToDB = storeImage;
  
/**
 * Get storage usage statistics for the database
 * @param {string} dbName - Database name
 * @returns {Object} - Statistics about storage usage
 */
export const getStorageStats = async (dbName = 'xrayImagesDB') => {
  const db = await openDatabase(dbName);
  return new Promise((resolve, reject) => {
    const tx = db.transaction('images', 'readonly');
    const store = tx.objectStore('images');
    const req = store.getAll();
    
    req.onsuccess = () => {
      const images = req.result;
      let totalSize = 0;
      let totalImages = images.length;
      let patientCount = new Set();
      let evaluatedCount = 0;
      let unusedCount = 0;
      let usedCount = 0;
      let purgedCount = 0;
      
      images.forEach(image => {
        // Calculate blob size if available
        if (image.imageBlob && image.imageBlob.size) {
          totalSize += image.imageBlob.size;
        }
        
        // Count unique patients
        if (image.patientId) {
          patientCount.add(image.patientId);
        }

        // Count evaluated images
        if (image.isEvaluated) {
          evaluatedCount++;
        }
        
        // Count by training status
        switch (image.trainingStatus) {
          case 'unused':
            unusedCount++;
            break;
          case 'used':
            usedCount++;
            break;
          case 'purged':
            purgedCount++;
            break;
        }
      });
      
      resolve({
        totalImages,
        evaluatedImages: evaluatedCount,
        unusedImages: unusedCount,
        usedImages: usedCount,
        purgedImages: purgedCount,
        totalSize,
        totalSizeMB: (totalSize / (1024 * 1024)).toFixed(2),
        uniquePatients: patientCount.size,
        averageImageSize: totalImages > 0 ? (totalSize / totalImages).toFixed(2) : 0
      });
    };
    
    req.onerror = (e) => reject(e.target.errorCode);
  });
};

