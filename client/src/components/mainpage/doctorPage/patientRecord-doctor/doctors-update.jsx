import React, { useState, useEffect } from 'react';
import add2 from '../../../../images/add2.png';
import axios from 'axios';

function Update({ patientId, xrayImages }) {
  const apiUrl = process.env.REACT_APP_API_BASE_URL || "http://localhost:3000";
  const [evaluation, setEvaluation] = useState('');
  const [findings, setFindings] = useState('');
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isClicked, setIsClicked] = useState(false);
  const [isEnlarged, setIsEnlarged] = useState(false);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isUpdated, setIsUpdated] = useState(false);

  // Load update status from localStorage
  useEffect(() => {
    const updatedStatus = localStorage.getItem(`updated-${patientId}`);
    setIsUpdated(updatedStatus === "true");
  }, [patientId]);

  const handleUpload = async () => {
    if (!patientId) {
      console.error("No patient selected.");
      return;
    }
    console.log("Selected Patient ID:", patientId);
    try {
      const response = await axios.put(
        `${apiUrl}/patients/updateEvaluation/${patientId}`,
        { evaluation, findings }
      );
      console.log("Evaluation updated successfully:", response.data);
      
      // Mark as updated in localStorage
      localStorage.setItem(`updated-${patientId}`, "true");
      setIsUpdated(true);

    } catch (error) {
      console.error("Error updating evaluation:", error);
    } finally {
      setIsModalOpen(false);
      setIsClicked(false);
    }
  };

  return (
    <div>
      <div>
        <img
          src={add2}
          alt="add2"
          className={`w-12 h-12 cursor-pointer mt-2 rounded-full transition-all ${
            isUpdated ? 'bg-white bg-opacity-50' : 'bg-[#00FF47] bg-opacity-30'
          }`}
          onClick={() => {
            setIsModalOpen(true);
            setIsClicked(true);
          }}
        />
      </div>

      {isModalOpen && (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
          <div className="bg-gray-800 text-white p-6 rounded-lg shadow-lg w-3/5 relative">
            <h2 className="text-lg font-bold mb-4 text-start">Doctor Evaluation</h2>

            {/* X-ray Carousel */}
            {xrayImages && xrayImages.length > 0 && (
              <div className="mb-4 relative flex flex-row justify-center">
                <img
                  src={`data:image/jpeg;base64,${xrayImages[currentIndex]}`}
                  alt={`X-ray ${currentIndex + 1}`}
                  className={`cursor-pointer rounded-md object-cover p-4 bg-slate-700 transition-all duration-300 ${
                    isEnlarged ? 'hidden' : 'w-90 h-80'
                  }`}
                  onClick={() => setIsEnlarged(true)}
                />

                {/* Enlarged image overlay */}
                {isEnlarged && (
                  <div 
                    className="fixed inset-0 flex items-center justify-center bg-gray-500 bg-opacity-75 z-50"
                    onClick={() => setIsEnlarged(false)}
                  >
                    <img 
                      src={`data:image/jpeg;base64,${xrayImages[currentIndex]}`}
                      alt={`Enlarged X-ray ${currentIndex + 1}`}
                      className="max-w-[90vw] max-h-[90vh] cursor-pointer"
                    />
                  </div>
                )}

                {xrayImages.length > 1 && !isEnlarged && (
                  <>
                    <button 
                      onClick={() => setCurrentIndex((prev) => (prev === 0 ? xrayImages.length - 1 : prev - 1))}
                      className="absolute top-1/2 left-0 transform -translate-y-1/2 bg-gray-600 text-white p-2 rounded-full"
                    >
                      ‹
                    </button>
                    <button 
                      onClick={() => setCurrentIndex((prev) => (prev === xrayImages.length - 1 ? 0 : prev + 1))}
                      className="absolute top-1/2 right-0 transform -translate-y-1/2 bg-gray-600 text-white p-2 rounded-full"
                    >
                      ›
                    </button>
                  </>
                )}
              </div>
            )}

            {/* Text area for evaluation */}
            <div className="mb-4 flex flex-row justify-center">
              <textarea
                className="w-3/5 p-3 rounded-md bg-gray-700 text-white focus:outline-none focus:ring-2 focus:ring-lime-500"
                placeholder="Type impression..."
                rows="1"
                value={evaluation}
                onChange={(e) => setEvaluation(e.target.value)}
              ></textarea>
            </div>

            <div className="mb-4 flex flex-row justify-center">
              <textarea
                className="w-3/5 p-3 rounded-md bg-gray-700 text-white focus:outline-none focus:ring-2 focus:ring-lime-500"
                placeholder="Type findings..."
                rows="4"
                value={findings}
                onChange={(e) => setFindings(e.target.value)}
              ></textarea>
            </div>
            
            <div className="flex justify-between items-center">
              <button 
                className="px-4 py-2 bg-gray-500 rounded-md hover:bg-gray-600" 
                onClick={() => setIsModalOpen(false)}
              >
                Cancel
              </button>
              <button 
                className="px-4 py-2 bg-[#23AE5B] text-white rounded-md hover:bg-[#23AE4C]" 
                onClick={handleUpload}
              >
                + Upload
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Update;
