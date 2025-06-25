import React, { useState } from 'react';
import axios from 'axios';
import { saveImageToDB } from '../../../utils/indexedDBUtils';

export default function DoctorsEvaluation({ evaluation, findings, xrayImages, patientId, patientInfo }) {
  const [isSaving, setIsSaving] = useState(false);
  const [saveStatus, setSaveStatus] = useState(null);
  const apiUrl = process.env.REACT_APP_API_BASE_URL || "http://localhost:3000";

  // Determine classification based on evaluation text
  const getClassification = () => {
    if (!evaluation) return 'Unclassified';
    
    const evalText = evaluation.toLowerCase();
    if (evalText.includes('normal')) return 'Normal';
    if (evalText.includes('fibrosis')) return 'Fibrosis';
    if (evalText.includes('edema')) return 'Edema';
    if (evalText.includes('covid')) return 'COVID-19';
    if (evalText.includes('pneumonia')) return 'Pneumonia';
    return 'Unclassified';
  };

  // Save evaluation and images to both server and IndexedDB
  const saveEvaluation = async () => {
    if (!evaluation || !patientId) return;
    
    setIsSaving(true);
    try {
      const classification = getClassification();
      const token = localStorage.getItem('token') || sessionStorage.getItem('token');
      
      // 1. Save to server if online
      if (navigator.onLine) {
        const evaluationData = {
          patientId,
          evaluation,
          findings,
          classification,
          trainingStatus: 'unused', // Set initial training status
          timestamp: new Date().toISOString()
        };
        
        await axios.post(`${apiUrl}/evaluations/save`, evaluationData, {
          headers: { 'Authorization': `Bearer ${token}` }
        });
      }
      
      // 2. Save to IndexedDB (with evaluation data attached)
      if (xrayImages && xrayImages.length > 0) {
        for (let i = 0; i < xrayImages.length; i++) {
          // Convert base64 to blob if needed
          let imageData = xrayImages[i];
          
          // Save with evaluation metadata
          await saveImageToDB(imageData, {
            id: `eval-${patientId}-${Date.now()}-${i}`,
            patientId,
            patientName: patientInfo?.name || 'Unknown',
            patientLocation: patientInfo?.location || 'Unknown',
            patientAge: patientInfo?.age || 'Unknown',
            patientGender: patientInfo?.gender || 'Unknown',
            evaluation,
            findings,
            classification,
            isEvaluated: true,
            evaluationTimestamp: Date.now(),
            imageIndex: i,
            // Add training status metadata
            trainingStatus: 'unused',
            lastTrainingDate: null
          });
        }
      }
      
      setSaveStatus('success');
    } catch (error) {
      console.error('Error saving evaluation:', error);
      setSaveStatus('error');
    } finally {
      setIsSaving(false);
      // Reset status after 3 seconds
      setTimeout(() => setSaveStatus(null), 3000);
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
          
          {/* Save Button */}
          
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
              Error saving evaluation. Please try again.
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
          
          {/* X-ray Images Gallery */}
          {xrayImages && xrayImages.length > 0 && (
            <div>
              <h3 className="text-lg font-semibold text-gray-700 mb-2">X-ray Images:</h3>
              <div className="grid grid-cols-2 gap-2">
                {xrayImages.map((image, index) => (
                  <div key={index} className="relative group">
                    <img
                      src={`data:image/jpeg;base64,${image}`}
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