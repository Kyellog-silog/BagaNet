import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { saveImageToDB, getStorageStats } from '../../../utils/indexedDBUtils';

export default function DoctorsEvaluation({ evaluation, findings, xrayImages, patientId, patientInfo }) {
  const [isSaving, setIsSaving] = useState(false);
  const [saveStatus, setSaveStatus] = useState(null);
  const [debugInfo, setDebugInfo] = useState(null);
  const apiUrl = process.env.REACT_APP_API_BASE_URL || "http://localhost:3000";

  useEffect(() => {
    const checkStorage = async () => {
      try {
        const stats = await getStorageStats();
        console.log('Current IndexedDB Stats:', stats);
      } catch (err) {
        console.error('Error checking storage stats:', err);
      }
    };
    
    checkStorage();
  }, []);

  const getClassification = () => {
    if (!evaluation) return 'Unclassified';
    
    const evalText = evaluation.toLowerCase();
    if (evalText.includes('normal')) return 'Normal';
    if (evalText.includes('pneumonia')) return 'Pneumonia';
    if (evalText.includes('edema')) return 'Edema';
    if (evalText.includes('covid')) return 'COVID-19';
    if (evalText.includes('pneumothorax')) return 'Pneumothorax';
    
    if (evalText.includes('abnormal') || 
        evalText.includes('opacity') || 
        evalText.includes('infiltrate') || 
        evalText.includes('consolidation')) {
      return 'Abnormal - Other';
    }
    
    return 'Unclassified';
  };

  const saveEvaluation = async () => {
    if (!evaluation) {
      setDebugInfo('Missing required data: evaluation');
      return;
    }
    
    setIsSaving(true);
    setDebugInfo(null);
    
    try {
      const classification = getClassification();
      const token = localStorage.getItem('token') || sessionStorage.getItem('token');
      const effectivePatientId = patientId || `unknown-${Date.now()}`;
      
      console.log('Saving evaluation with data:', {
        patientId: effectivePatientId,
        evaluation,
        findings,
        classification,
        imageCount: xrayImages?.length || 0
      });
      
      
      if (xrayImages && xrayImages.length > 0) {
        const savedIds = [];
        
        for (let i = 0; i < xrayImages.length; i++) {
          try {
            let imageData = xrayImages[i];
            
            if (typeof imageData === 'string' && !imageData.startsWith('data:')) {
              imageData = `data:image/jpeg;base64,${imageData}`;
            }
            
            const imageId = `eval-${effectivePatientId}-${Date.now()}-${i}`;
            
            const metadata = {
              id: imageId,
              patientId: effectivePatientId,
              evaluation,
              findings,
              classification,
              isEvaluated: true,
              timestamp: Date.now(),
              trainingStatus: 'unused'
            };
            
            if (patientInfo) {
              if (patientInfo.name) metadata.patientName = patientInfo.name;
              if (patientInfo.age) metadata.patientAge = patientInfo.age;
              if (patientInfo.gender) metadata.patientGender = patientInfo.gender;
            }
            
            console.log(`Saving image ${i+1} with classification: ${classification}`);
            const savedId = await saveImageToDB(imageData, metadata);
            console.log(`Successfully saved image ${i+1} with ID:`, savedId);
            savedIds.push(savedId);
          } catch (imgError) {
            console.error(`Error saving image ${i+1}:`, imgError);
            setDebugInfo(prev => `${prev || ''}\nImage ${i+1} error: ${imgError.message}`);
          }
        }
        
        console.log(`Saved ${savedIds.length}/${xrayImages.length} images to IndexedDB`);
        if (savedIds.length === 0) {
          throw new Error('Failed to save any images to IndexedDB');
        }
      } else {
        console.warn('No X-ray images provided to save');
        setDebugInfo('No X-ray images provided to save');
      }
            const updatedStats = await getStorageStats();
      console.log('Updated IndexedDB Stats after save:', updatedStats);
      
      setSaveStatus('success');
    } catch (error) {
      console.error('Error saving evaluation:', error);
      setSaveStatus('error');
      setDebugInfo(`Save Error: ${error.message}`);
    } finally {
      setIsSaving(false);
      setTimeout(() => setSaveStatus(null), 5000);
    }
  };

  return (
    <div className="p-6 rounded-lg bg-white shadow-md">
      <h2 className="text-2xl font-bold text-gray-800 mb-6 border-b pb-2">Radiology Report</h2>
      
      <div className="flex flex-col md:flex-row gap-6">
        {/* Left Column - Evaluation Details */}
        <div className="w-full md:w-2/3 flex flex-col">
          {/* Impression Section */}
          <div className="mb-6">
            <h3 className="text-lg font-semibold text-gray-700 mb-2">Impression:</h3>
            <div className="p-4 bg-gray-50 rounded-md border border-gray-200">
              {evaluation ? (
                <p className="text-gray-800">{evaluation}</p>
              ) : (
                <p className="text-gray-500 italic">Not yet evaluated by the doctor</p>
              )}
            </div>
          </div>
          
          {/* Findings Section */}
          <div className="mb-6">
            <h3 className="text-lg font-semibold text-gray-700 mb-2">Findings:</h3>
            <div className="p-4 bg-gray-50 rounded-md border border-gray-200">
              {findings ? (
                <p className="text-gray-800">{findings}</p>
              ) : (
                <p className="text-gray-500 italic">No findings recorded</p>
              )}
            </div>
          </div>
          
          {/* Debug Info (only shown when there's debug info) */}
          {debugInfo && (
            <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-md text-xs font-mono overflow-auto max-h-40">
              <h4 className="font-bold mb-1">Debug Info:</h4>
              <pre>{debugInfo}</pre>
            </div>
          )}
          
          {/* Save Button - Only require evaluation */}
          {evaluation && !isSaving && !saveStatus && (
            <button 
              onClick={saveEvaluation}
              className="mt-4 bg-green-500 hover:bg-green-600 text-white py-2 px-4 rounded-md transition"
            >
              Save Evaluation & Images
            </button>
          )}
          
          {/* Loading State */}
          {isSaving && (
            <div className="mt-4 flex items-center text-blue-600">
              <svg className="animate-spin h-5 w-5 mr-2" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Saving evaluation and images...
            </div>
          )}
          
          {/* Success/Error Message */}
          {saveStatus === 'success' && (
            <div className="mt-4 bg-green-100 border border-green-300 text-green-700 px-4 py-2 rounded-md">
              Evaluation and images saved successfully! Images marked for training.
            </div>
          )}
          
          {saveStatus === 'error' && (
            <div className="mt-4 bg-red-100 border border-red-300 text-red-700 px-4 py-2 rounded-md">
              Error saving evaluation. Please try again or check console for details.
            </div>
          )}
        </div>
        
        {/* Right Column - Images and Classification Display */}
        <div className="w-full md:w-1/3 flex flex-col">
          {/* Classification Display */}
          <div className="mb-4">
            <h3 className="text-lg font-semibold text-gray-700 mb-2">Classification:</h3>
            <div className="p-3 bg-blue-50 rounded-md border border-blue-200 text-blue-800 font-medium text-center">
              {getClassification()}
            </div>
          </div>
          
          {/* Training Status */}
          <div className="mb-4">
            <h3 className="text-lg font-semibold text-gray-700 mb-2">Training Status:</h3>
            <div className="p-3 bg-yellow-50 rounded-md border border-yellow-200 text-yellow-800 font-medium text-center">
              Unused (Ready for Training)
            </div>
          </div>
          
          {/* Patient ID display (if available) */}
          {patientId && (
            <div className="mb-4">
              <h3 className="text-lg font-semibold text-gray-700 mb-2">Patient ID:</h3>
              <div className="p-3 bg-gray-50 rounded-md border border-gray-200 text-gray-800 font-medium text-center">
                {patientId}
              </div>
            </div>
          )}
          
          {/* X-ray Images Gallery */}
          {xrayImages && xrayImages.length > 0 && (
            <div>
              <h3 className="text-lg font-semibold text-gray-700 mb-2">X-ray Images:</h3>
              <div className="grid grid-cols-2 gap-2">
                {xrayImages.map((image, index) => (
                  <div key={index} className="relative group">
                    <img
                      src={typeof image === 'string' && !image.startsWith('data:') 
                        ? `data:image/jpeg;base64,${image}` 
                        : image}
                      alt={`X-ray ${index + 1}`}
                      className="w-full h-32 object-cover rounded-md border border-gray-300"
                    />
                    <span className="absolute bottom-1 right-1 bg-black bg-opacity-50 text-white text-xs px-2 py-1 rounded">
                      #{index + 1}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}