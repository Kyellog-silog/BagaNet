import Sidebar from '../dashboard-radtech/sidebar';
import Navbar from '../dashboard-radtech/navbar';
import search from '../../../../images/search.png';
import React, { useEffect, useState, useCallback } from 'react';
import axios from 'axios';
import { AiOutlineCheckCircle, AiOutlineCloudUpload } from 'react-icons/ai';
import { MdClear } from 'react-icons/md';
import Update from './update';
import Delete from './delete';
import View from './view';
import { 
  saveImageToDB, 
  getAllStoredImages, 
  getImagesByPatientId, 
  revokeObjectURLs,
  base64ToBlob,
  storeExists,
  getModel,
  storeModel
} from '../../../utils/indexedDBUtils';

const getAuthToken = async (clientId = "web-client") => {
  const modelServerUrl = process.env.REACT_APP_MODEL_SERVER_URL || "http://localhost:3000";
  const apiKey = process.env.REACT_APP_MODEL_API_KEY || "FeDMl2025";
  
  try {
    const response = await axios.post(`${modelServerUrl}/api/token`, 
      { client_id: clientId },
      { 
        headers: { 
          'X-API-Key': apiKey,
          'X-Client-ID': clientId
        }
      }
    );
    
    // Store token in localStorage
    localStorage.setItem('fedml_token', response.data.access_token);
    
    return response.data.access_token;
  } catch (error) {
    console.error('Error getting FedML auth token:', error);
    throw error;
  }
};


const PatientActivity = () => {
  const apiUrl = process.env.REACT_APP_API_BASE_URL || "http://localhost:3000";
  const modelServerUrl = process.env.REACT_APP_MODEL_SERVER_URL || "http://localhost:5050";
  const [patients, setPatients] = useState([]);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    location: '',
    age: '',
    gender: '',
  });
  const [files, setFiles] = useState([]);
  const [filePreviewUrls, setFilePreviewUrls] = useState([]);
  const [submittedImage, setSubmittedImage] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [validationErrors, setValidationErrors] = useState({});
  const [isOffline, setIsOffline] = useState(!navigator.onLine);
  const [searchTerm, setSearchTerm] = useState('');
  const [grabStatus, setGrabStatus] = useState('');
  const apiKey = process.env.REACT_APP_MODEL_API_KEY || "FeDMl2025";
  const [authToken, setAuthToken] = useState(null);
  

  // Check for network status changes
  useEffect(() => {
    const handleOnline = () => setIsOffline(false);
    const handleOffline = () => setIsOffline(true);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  useEffect(() => {
    if (navigator.onLine) {
      fetchPatients();
    } else {
      loadPatientsFromIndexedDB();
    }
    
    // Cleanup function to revoke URLs when component unmounts
    return () => {
      filePreviewUrls.forEach(url => URL.revokeObjectURL(url));
    };
  }, [isOffline]);



  
  useEffect(() => {
    if (!localStorage.getItem('modelVersion')) localStorage.setItem('modelVersion', '0');
  }, []);

  const fetchPatients = async () => {
    try {
      const token = localStorage.getItem('token') || sessionStorage.getItem('token');
      if (!token) {
        console.error('No authentication token found');
        return;
      }
  
      const response = await axios.get(`${apiUrl}/patients/patients`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
  
      // Convert base64 X-ray images to object URLs for display only
      const processedPatients = await Promise.all(response.data.map(async (patient) => {
        if (patient.xray && patient.xray.length > 0) {
          const objectURLs = await Promise.all(patient.xray.map(async (xrayImage) => {
            const blob = base64ToBlob(xrayImage);
            return URL.createObjectURL(blob);
          }));
          return { ...patient, xrayObjectURLs: objectURLs };
        }
        return patient;
      }));
  
      setPatients(processedPatients);
    } catch (error) {
      console.error('Error fetching patients:', error);
      setError('Failed to fetch patients');
      // Optionally remove loadPatientsFromIndexedDB() here too
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleFileChange = (event) => {
    const selectedFiles = event.target.files;
    if (selectedFiles && selectedFiles.length > 0) {
      const newFiles = Array.from(selectedFiles);
      setFiles(prevFiles => [...prevFiles, ...newFiles]);
      
      // Create preview URLs for the selected files
      const newPreviewUrls = Array.from(selectedFiles).map(file => URL.createObjectURL(file));
      setFilePreviewUrls(prevUrls => [...prevUrls, ...newPreviewUrls]);
    }
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const droppedFiles = event.dataTransfer.files;
    if (droppedFiles.length > 0) {
      const newFiles = Array.from(droppedFiles);
      setFiles(prevFiles => [...prevFiles, ...newFiles]);
      
      // Create preview URLs for the dropped files
      const newPreviewUrls = Array.from(droppedFiles).map(file => URL.createObjectURL(file));
      setFilePreviewUrls(prevUrls => [...prevUrls, ...newPreviewUrls]);
    }
  };

  const loadPatientsFromIndexedDB = async () => {
    try {
      // First try to get evaluated images
      const evaluatedImages = await getEvaluatedImages();
      
      // Then get all images as fallback
      const allImages = evaluatedImages.length > 0 ? 
        evaluatedImages : 
        await getAllStoredImages();
      
      // Group images by patient ID to recreate patient structure
      const patientsMap = new Map();
      
      allImages.forEach(item => {
        if (!item.patientId) return;
        
        if (!patientsMap.has(item.patientId)) {
          patientsMap.set(item.patientId, {
            _id: item.patientId,
            name: item.patientName || 'Unknown',
            location: item.patientLocation || 'Unknown',
            age: item.patientAge || 'Unknown',
            gender: item.patientGender || 'Unknown',
            xray: [],
            evaluation: item.evaluation || null,
            findings: item.findings || null,
            classification: item.classification || null,
            isOffline: item.pendingUpload || false,
            isEvaluated: item.isEvaluated || false
          });
        }
        
        // Add image to patient's xray array
        const patient = patientsMap.get(item.patientId);
        if (item.imageBlob && item.objectURL) {
          patient.xray.push(item.objectURL);
          
          // Make sure we capture evaluation data if it exists
          if (item.isEvaluated && item.evaluation) {
            patient.evaluation = item.evaluation;
            patient.findings = item.findings || null;
            patient.classification = item.classification || 'Unclassified';
            patient.isEvaluated = true;
          }
        }
      });
      
      setPatients(Array.from(patientsMap.values()));
    } catch (error) {
      console.error('Error loading patients from IndexedDB:', error);
      setError('Failed to load offline patient data');
    }
  };

  const closeModal = () => {
    setIsModalOpen(false);
    setFormData({ name: '', location: '', age: '', gender: '' });
    setFiles([]);
    setSubmittedImage(null);
    setValidationErrors({});
    setError(null);
  };

const handleGrabModel = async () => {
  // 0) Dynamically import TFJS & pick a backend
  const tf = await import('@tensorflow/tfjs');
  await import('@tensorflow/tfjs-backend-wasm');
  const preferred = ['wasm', 'cpu'];
  let actualBackend = null;
  for (const b of preferred) {
    try {
      await tf.setBackend(b);
      await tf.ready();
      console.log(`✅ Initialized TFJS backend "${b}"`);
      actualBackend = b;
      break;
    } catch (e) {
      console.warn(`❌ Could not initialize TFJS backend "${b}"`, e);
    }
  }
  if (!actualBackend) {
    throw new Error('No usable TFJS backend available');
  }

  try {
    // 1) IndexedDB setup for ONNX & TFJS
    const dbName = 'xrayImagesDB';
    const token = 'dummy-token-for-development'; // bypass auth in dev
    console.log('⚠️ Using dummy token (token check bypassed)');
    const models = await tf.io.listModels();
    console.log('Available models:', models);

    const hasModelsStore = await storeExists('models', dbName);
    if (!hasModelsStore) {
      console.log('Creating models store in IndexedDB...');
      await openDatabase(dbName);
    }

    // 2) ONNX: fetch & cache if missing
    const onnxBlob = await getModel('onnx_default', dbName);
    if (!onnxBlob) {
      console.log('⬇️ Downloading missing ONNX model...');
      const onnxResponse = await fetch(`${modelServerUrl}/model`, {
        method: 'GET',
        mode: 'cors',
        credentials: 'omit',
        headers: {
          'X-API-Key': apiKey,
          'Authorization': `Bearer ${token}`,
          'ngrok-skip-browser-warning': 'true',
        }
      });
      if (!onnxResponse.ok) {
        const txt = await onnxResponse.text();
        console.error('ONNX Response Error:', { status: onnxResponse.status, preview: txt.substring(0,200) });
        throw new Error(`ONNX download failed: ${onnxResponse.status}`);
      }
      const buf = await onnxResponse.arrayBuffer();
      await storeModel(buf, 'onnx_default', dbName);
      console.log('✅ ONNX model saved');
    } else {
      console.log('✅ ONNX model already in IndexedDB');
    }

    // 3) TFJS: LayersModel & GraphModel download & cache
    //    Only fetch if not already in IndexedDB
    const layersKey = 'indexeddb://tfjs_layers_default';
    const graphKey  = 'indexeddb://tfjs_graph_default';
    const hasLayers = Boolean(models[layersKey]);
    const hasGraph  = Boolean(models[graphKey]);

    // URLs to your static assets
    const version   = 'v1';
    const basePath  = `${modelServerUrl}/static/tfjs/${version}`;
    const layersUrl = `${basePath}/model.json`;
    const graphUrl  = `${basePath}/graph/model.json`;

    // Fetch & save LayersModel
    if (!hasLayers) {
      console.log(`⬇️ Loading LayersModel from ${layersUrl}`);
      const layersModel = await tf.loadLayersModel(layersUrl);
      await layersModel.save(layersKey);
      console.log('✅ LayersModel saved to IndexedDB');
    } else {
      console.log('✅ LayersModel already in IndexedDB');
    }

    // Fetch & save GraphModel
    if (!hasGraph) {
      console.log(`⬇️ Loading GraphModel from ${graphUrl}`);
      const graphModel = await tf.loadGraphModel(graphUrl);
      await graphModel.save(graphKey);
      console.log('✅ GraphModel saved to IndexedDB');
    } else {
      console.log('✅ GraphModel already in IndexedDB');
    }

    setGrabStatus('All models ready in IndexedDB');
  } catch (err) {
    console.error('Error in handleGrabModel:', err);
    setGrabStatus(`Error: ${err.message}`);
    if (err.message.includes('HTML')) {
      console.error('Authentication issue detected—received HTML instead of JSON');
    }
  }
};

  const handleRemoveFile = (index) => {
    // Revoke the object URL to prevent memory leaks
    if (filePreviewUrls[index]) {
      URL.revokeObjectURL(filePreviewUrls[index]);
    }
    
    setFiles(prevFiles => prevFiles.filter((_, i) => i !== index));
    setFilePreviewUrls(prevUrls => prevUrls.filter((_, i) => i !== index));
  };

  const validateForm = () => {
    const errors = {};
    if (!formData.name.trim()) errors.name = 'Name is required';
    if (!formData.location.trim()) errors.location = 'Location is required';
    if (!formData.age) errors.age = 'Age is required';
    if (!formData.gender) errors.gender = 'Gender is required';
    if (files.length === 0) errors.files = 'At least one X-ray image is required';
    
    files.forEach((file, index) => {
      if (!file.type.startsWith('image/')) {
        errors[`file${index}`] = 'Only image files are allowed';
      }
      if (file.size > 5 * 1024 * 1024) { // 5MB limit
        errors[`file${index}`] = 'File size must be less than 5MB';
      }
    });

    setValidationErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setValidationErrors({});

    if (!validateForm()) {
      return;
    }

    setIsSubmitting(true);

    try {
      if (navigator.onLine) {
        const token = localStorage.getItem('token') || sessionStorage.getItem('token');
        if (!token) {
          throw new Error('No authentication token found');
        }
    
        const formDataToSend = new FormData();
        Object.keys(formData).forEach(key => {
          formDataToSend.append(key, formData[key]);
        });
        files.forEach(file => {
          formDataToSend.append('xray', file);
        });
    
        const response = await axios.post(`${apiUrl}/patients/addPatient`, formDataToSend, {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'multipart/form-data'
          }
        });
    
        console.log('Server response:', response.data);
    
        // Set submitted image preview
        if (files.length > 0 && filePreviewUrls.length > 0) {
          setSubmittedImage(filePreviewUrls[0]);
        }
    
        // Refresh patient list
        fetchPatients();
      } else {
        // You can optionally alert or block submission if offline
        setError("You are offline. Please connect to the internet to upload.");
        return;
      }
    
      // Clear form
      setIsModalOpen(false);
      setFormData({ name: '', location: '', age: '', gender: '' });
      setFiles([]);
      setFilePreviewUrls([]);
      closeModal();
    
    } catch (error) {
      console.error('Error uploading files:', error);
      const errorMessage = error.response?.data?.message || 'Failed to add patient';
      setError(errorMessage);
      console.error('Detailed error:', {
        message: error.message,
        response: error.response?.data,
        status: error.response?.status
      });
    } finally {
      setIsSubmitting(false);
    }
    
  };



  const handlePatientDeletion = (patientId) => {
    setPatients(patients.filter(patient => patient._id !== patientId));
  };

  const handleSearchChange = (e) => {
    setSearchTerm(e.target.value);
  };

  const filteredPatients = patients.filter(patient => 
    patient.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    patient.location.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="flex h-screen bg-gray-100">
      <Sidebar />
      <div className="flex flex-col flex-grow">
        <Navbar />
        <div className="bg-white p-6 rounded-lg shadow-md mt-4 mx-6">
          <h3 className="text-4xl font-bold mb-6 text-gray-800">Patient</h3>
          
          {isOffline && (
            <div className="mb-4 p-3 bg-yellow-100 text-yellow-800 rounded-md flex items-center">
              <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
              You are currently offline. Some features may be limited.
            </div>
          )}

          <div className="flex flex-row justify-between mb-4">
            <div className="flex items-center -ml-2">
              <input
                type="text"
                placeholder="Search"
                value={searchTerm}
                onChange={handleSearchChange}
                aria-label="Search Bar"
                className="border rounded-md px-4 py-2 w-96 ml-2 border-gray-300 focus:outline-none focus:ring-1 bg-[#E3EBF3] focus:ring-blue-500"
              />
              <img src={search} alt="Search icon" className="w-5 h-5 -ml-8 cursor-pointer mt-2" />
            </div>
            <div className="flex flex-row ml-6">
              <div>
                <div className="flex flex-row gap-2">
                  <div
                  className="p-2 px-6 text-base text-white bg-lime-500 rounded-lg hover:bg-lime-600 cursor-pointer"
                  onClick={() => setIsModalOpen(true)}
                >
                  Add Patient
                </div>

                <div
                  className="p-2 px-6 text-base text-white bg-blue-500 rounded-lg hover:bg-blue-600 cursor-pointer"
                  onClick={() => {
                    handleGrabModel();
                  }}
                >
                  Grab Model
                </div>
                </div>
                
                
                {isModalOpen && (
                  <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
                    <div className="bg-gray-800 p-6 rounded-lg shadow-lg w-96">
                      <h2 className="text-xl font-bold mb-4 text-white">Add Patient Details</h2>
                      <div className="mb-4">
                        <label className="block text-sm font-medium text-white">Patient Name</label>
                        <input
                          type="text"
                          name="name"
                          value={formData.name}
                          onChange={handleInputChange}
                          className={`mt-1 p-2 w-full border rounded-lg bg-gray-700 border-none ${
                            validationErrors.name ? 'border-red-500' : ''
                          }`}
                        />
                        {validationErrors.name && (
                          <p className="text-red-500 text-sm mt-1">{validationErrors.name}</p>
                        )}
                      </div>
                      <div className="mb-4">
                        <label className="block text-sm font-medium text-white">Location</label>
                        <input
                          type="text"
                          name="location"
                          value={formData.location}
                          onChange={handleInputChange}
                          className={`mt-1 p-2 w-full border rounded-lg bg-gray-700 border-none ${
                            validationErrors.location ? 'border-red-500' : ''
                          }`}
                        />
                        {validationErrors.location && (
                          <p className="text-red-500 text-sm mt-1">{validationErrors.location}</p>
                        )}
                      </div>
                      <div className="mb-4">
                        <label className="block text-sm font-medium text-white">Age</label>
                        <input
                          type="number"
                          name="age"
                          value={formData.age}
                          onChange={handleInputChange}
                          className={`mt-1 p-2 w-full border rounded-lg bg-gray-700 border-none ${
                            validationErrors.age ? 'border-red-500' : ''
                          }`}
                        />
                        {validationErrors.age && (
                          <p className="text-red-500 text-sm mt-1">{validationErrors.age}</p>
                        )}
                      </div>
                      <div className="mb-4">
                        <label className="block text-sm font-medium text-white">Gender</label>
                        <select
                          name="gender"
                          value={formData.gender}
                          onChange={handleInputChange}
                          className={`mt-1 p-2 w-full border rounded-lg bg-gray-700 text-white border-none ${
                            validationErrors.gender ? 'border-red-500' : ''
                          }`}
                        >
                          <option value="">Select Gender</option>
                          <option value="Male">Male</option>
                          <option value="Female">Female</option>
                          <option value="Other">Other</option>
                        </select>
                        {validationErrors.gender && (
                          <p className="text-red-500 text-sm mt-1">{validationErrors.gender}</p>
                        )}
                        
                      </div>

                      <section className="py-2 w-full h-64">
                        {submittedImage ? (
                          <div className="flex flex-col items-center">
                            <img
                              src={submittedImage}
                              alt="Submitted Preview"
                              className="w-full max-w-xs h-auto rounded-md mb-4"
                            />
                            <p className="text-green-600 font-bold">Image submitted successfully!</p>
                          </div>
                        ) : (
                          <div
                            className={`border-2 border-dashed rounded-lg py-4 text-center flex flex-col items-center justify-center ${
                              validationErrors.files ? 'border-red-500' : ''
                            }`}
                            onDrop={handleDrop}
                            onDragOver={(event) => event.preventDefault()}
                          >
                            <div className="flex flex-col items-center mb-4 text-gray-600">
                              <AiOutlineCloudUpload className="text-xl mb-2" />
                              <p className="text-xs font-bold">Drag and drop images here</p>
                            </div>
                            <input
                              type="file"
                              hidden
                              id="browse"
                              onChange={handleFileChange}
                              accept="image/*"
                              multiple
                            />
                            <label
                              htmlFor="browse"
                              className="text-xs bg-blue-500 text-white px-4 py-2 rounded-md cursor-pointer hover:bg-blue-400 transition"
                            >
                              Browse files
                            </label>
                            {validationErrors.files && (
                              <p className="text-red-500 text-sm mt-2">{validationErrors.files}</p>
                            )}
                          </div>
                        )}

                        {filePreviewUrls.length > 0 && (
                          <div className="flex mt-4 overflow-x-auto space-x-2">
                            {filePreviewUrls.map((url, index) => (
                              <div key={index} className="relative">
                                <img 
                                  src={url} 
                                  alt={`Preview ${index}`}
                                  className="h-16 w-16 object-cover rounded"
                                />
                                <button
                                  className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full p-1 w-5 h-5 flex items-center justify-center"
                                  onClick={() => handleRemoveFile(index)}
                                >
                                  ×
                                </button>
                              </div>
                            ))}
                          </div>
                        )}

                        {files.length > 0 && (
                          <div className="w-full mt-4 overflow-y-auto border-t pt-4">
                            {files.map((file, index) => (
                              <div className="flex items-center justify-between p-2 border-b" key={index}>
                                <p className="text-xs font-medium text-white">{file.name}</p>
                                <MdClear
                                  className="text-lg text-gray-500 hover:text-red-500 cursor-pointer"
                                  onClick={() => handleRemoveFile(index)}
                                />
                              </div>
                            ))}
                          </div>
                        )}

                        {files.length > 0 && (
                          <div className="flex items-center text-green-600 mt-2">
                            <AiOutlineCheckCircle className="mr-1" />
                            <p className="text-xs font-bold">{files.length} image(s) selected</p>
                          </div>
                        )}
                      </section>

                      {files.length > 0 && (
                        <button
                          onClick={handleSubmit}
                          disabled={isSubmitting}  
                          className={`mt-8 text-white text-xs px-4 py-2 rounded-md cursor-pointer transition ${isSubmitting ? 'bg-gray-400' : 'bg-green-500 hover:bg-green-400'}`}
                        >
                          {isSubmitting ? 'Submitting...' : 'Submit'}
                        </button>
                      )}

                      <div className="flex justify-end mt-4">
                        <button
                          className="px-4 py-2 bg-gray-300 rounded-lg hover:bg-gray-400"
                          onClick={() => {
                            setIsModalOpen(false);
                            // Clean up preview URLs when closing modal
                            filePreviewUrls.forEach(url => URL.revokeObjectURL(url));
                            setFilePreviewUrls([]);
                          }}
                        >
                          Cancel
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="overflow-x-auto mt-6 max-h-[60vh]">
            <table className="table-auto border-collapse w-full">
              <thead className="bg-gray-200 sticky top-0 z-10">
                <tr className="bg-gray-200 text-left text-gray-700">
                  <th className="px-6 py-2 text-start">Patient Name</th>
                  <th className="px-6 py-2 text-start">Location</th>
                  <th className="px-6 py-2 text-start">Gender</th>
                  <th className="px-6 py-2 text-start">Age</th>
                  <th className="px-6 py-2 text-start">Xray</th>
                  <th className="px-6 py-2 text-start">Status</th>
                  <th className="px-6 py-2 text-start">Action</th>
                </tr>
              </thead>
              <tbody className="text-sm text-gray-700">
                {filteredPatients.map((patient) => (
                  <tr key={patient._id} className={patient.isOffline ? 'bg-gray-100' : ''}>
                    <td className="px-6 py-2">{patient.name}</td>
                    <td className="px-6 py-2">{patient.location}</td>
                    <td className="px-6 py-2">{patient.gender}</td>
                    <td className="px-6 py-2">{patient.age}</td>
                    <td className="px-6 py-2"> 
                      {patient.xray && patient.xray.length > 0 ? (
                        <div className="flex flex-wrap space-x-2">
                          {patient.xray.map((xrayImage, index) => (
                            <img
                              key={index}
                              src={xrayImage.startsWith('data:') ? xrayImage : `data:image/jpeg;base64,${xrayImage}`}
                              alt={`X-ray ${index}`}
                              className="2xl:w-40 2xl:h-40 lg:w-20 lg:h-20 object-cover rounded-md"
                            />
                          ))}
                        </div>
                      ) : (
                        <p>No X-ray images available</p>
                      )}
                    </td>
                    <td className="px-6 py-2">
                      {patient.isOffline ? (
                        <span className="px-2 py-1 bg-yellow-100 text-yellow-800 rounded-md text-xs">
                          Pending Upload
                        </span>
                      ) : (
                        <span className="px-2 py-1 bg-green-100 text-green-800 rounded-md text-xs">
                          Synced
                        </span>
                      )}
                    </td>
                    <td className="px-6 py-2 flex space-x-2">
                      <Update patientId={patient._id} isOffline={patient.isOffline} />
                      <Delete patientId={patient._id} onDelete={handlePatientDeletion} />
                      <View patient={patient} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {error && (
            <div className="mt-4 p-3 bg-red-500 text-white rounded-md">
              {error}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PatientActivity;