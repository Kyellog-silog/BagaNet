import React, { useState } from 'react';
import view from '../../../../images/view.png';
import DoctorsEvaluation from './doctorsEvaluation';
import BAGANETEvaluation from './baga.netEvaluation';

function View({ patient }) { // Now receives a patient prop
  const [isDoctorOpen, setIsDoctorOpen] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isClicked, setIsClicked] = useState(false);

  const openModal = () => {
    setIsModalOpen(true);
    setIsClicked(true);
  };

  const closeModal = () => {
    setIsModalOpen(false);
    setIsClicked(false);
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

            {/* Tab Content */}
            <div className="py-6">
              {isDoctorOpen ? (
                <DoctorsEvaluation 
                  evaluation={patient?.evaluation} 
                  findings={patient?.findings}
                  xrayImages={patient?.xray} 
                />
              ) : (
                <BAGANETEvaluation 
                  patientId={patient?._id} 
                  xrayImages={patient?.xray} 
                />

              )}
            </div>

            {/* Modal Footer */}
            <div className="flex justify-between items-center mt-4">
              <button className="px-4 py-2 bg-gray-500 rounded-md hover:bg-gray-600" onClick={closeModal}>
                Back
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default View;
