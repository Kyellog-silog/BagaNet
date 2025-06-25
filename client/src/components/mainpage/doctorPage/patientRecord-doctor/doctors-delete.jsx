import React, { useState } from 'react';
import del2 from '../../../../images/del2.png';
import axios from 'axios';

function Delete({ onDelete, patientId }) {
  const apiUrl = process.env.REACT_APP_API_BASE_URL || "http://localhost:3000";
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isClicked, setIsClicked] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [isDeleted, setIsDeleted] = useState(false);  

  const openModal = () => {
    setIsModalOpen(true);
    setIsClicked(true);
  };

  const closeModal = () => {
    setIsModalOpen(false);
    setIsClicked(false);
  };

  const handleDelete = async () => {
    try {
      const token = localStorage.getItem('token') || sessionStorage.getItem('token');
      if (!token) {
        throw new Error('No authentication token found');
      }

      const response = await axios.delete(`${apiUrl}/patients/deletePatient/${patientId}`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (response.status === 200) {
        console.log(response.data.message); 
        setIsDeleted(true);
        closeModal();
        onDelete();
      }
    } catch (error) {
      console.error("Error deleting patient:", error);
      const errorMessage = error.response?.data?.error || 
                          error.response?.data?.details || 
                          "Failed to delete patient. Please try again.";
      setErrorMessage(errorMessage);
    }
  };

  return (
    <div>
      <img
        src={del2}
        alt="Delete Icon"
        className={`w-12 h-12 cursor-pointer mt-2 rounded-full transition-all ${isClicked ? 'bg-[#00FF47] bg-opacity-30' : ''}`}
        onClick={openModal}
      />

      {isModalOpen && (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
          <div className="bg-gray-800 text-white p-6 rounded-lg shadow-lg w-96">
            <h2 className="text-lg font-bold mb-4 text-center">
              Are you sure you want to delete this?
            </h2>

            <div className="flex justify-between items-center mt-10">
              <button
                className="px-4 py-2 bg-gray-500 rounded-md hover:bg-gray-600"
                onClick={closeModal}
              >
                Cancel
              </button>
              <button
                className="px-4 py-2 bg-red-500 text-white rounded-md hover:bg-red-600"
                onClick={handleDelete}
              >
                Yes
              </button>
            </div>
            {errorMessage && <p className="text-red-500 text-center mt-4">{errorMessage}</p>}
            {isDeleted && <p className="text-green-500 text-center mt-4">Patient deleted successfully!</p>}
          </div>
        </div>
      )}
    </div>
  );
}

export default Delete;
