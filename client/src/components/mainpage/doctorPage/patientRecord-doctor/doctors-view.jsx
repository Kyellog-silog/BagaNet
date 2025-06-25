import React, { useState, useEffect } from 'react';
import view from '../../../../images/view.png';
import Doctorsevaluation from './doctors-evaluation';
import BAGANETEvaluation from './doctors-baga.netEvaluation';
import axios from 'axios';

function View({ id }) {
  const apiUrl = process.env.REACT_APP_API_BASE_URL || "http://localhost:3000";
  const [isDoctorOpen, setIsDoctorOpen] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isClicked, setIsClicked] = useState(false);
  const [patientData, setPatientData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchPatientData = async () => {
    try {
      setLoading(true);
      setError(null);
      const token = localStorage.getItem('token') || sessionStorage.getItem('token');
      if (!token) {
        throw new Error('No authentication token found');
      }

      const response = await axios.get(`${apiUrl}/patients/patients/${id}`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (response.data) {
        setPatientData({
          ...response.data,
          evaluation: response.data.evaluation || '',
          findings: response.data.findings || ''
        });
      }
    } catch (error) {
      console.error('Error fetching patient data:', error);
      setError(error.response?.data?.message || 'Failed to fetch patient data');
    } finally {
      setLoading(false);
    }
  };

  const openModal = () => {
    setIsModalOpen(true);
    setIsClicked(true);
    fetchPatientData();
  };

  const closeModal = () => {
    setIsModalOpen(false);
    setIsClicked(false);
    setPatientData(null);
  };

  return (
    <div>
      <div>
        <img
          src={view}
          alt="view"
          className={`w-12 h-12 cursor-pointer mt-2 rounded-full transition-all ${isClicked ? 'bg-[#00FF47] bg-opacity-30' : ''}`}
          onClick={openModal}
        />
      </div>

      {isModalOpen && (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
          <div className="bg-gray-800 text-white p-6 rounded-lg shadow-lg w-3/5">
            <h2 className="text-lg font-bold mb-4 text-center">View Result</h2>

            {/* Tabs */}
            <div className="flex flex-row bg-gray-800 border-y border-gray-500">
              <div
                className={`p-2 px-20 cursor-pointer ${isDoctorOpen ? 'text-white bg-blue-600' : 'text-gray-400'}`}
                onClick={() => setIsDoctorOpen(true)}
              >
                Doctor's Evaluation
              </div>
              <div
                className={`p-2 px-20 cursor-pointer ${!isDoctorOpen ? 'text-white bg-blue-600' : 'text-gray-400'}`}
                onClick={() => setIsDoctorOpen(false)}
              >
                BAGA.NET Evaluation
              </div>
            </div>

            {/* Content */}
            <div className="mt-4">
              {loading ? (
                <p className="text-center">Loading...</p>
              ) : error ? (
                <p className="text-center text-red-500">{error}</p>
              ) : (
                <>
                  {isDoctorOpen ? (
                    <Doctorsevaluation 
                      evaluation={patientData?.evaluation} 
                      findings={patientData?.findings}
                    />
                  ) : (
                    <BAGANETEvaluation />
                  )}
                </>
              )}
            </div>

            {/* Close Button */}
            <div className="mt-4 flex justify-end">
              <button
                className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700"
                onClick={closeModal}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default View;