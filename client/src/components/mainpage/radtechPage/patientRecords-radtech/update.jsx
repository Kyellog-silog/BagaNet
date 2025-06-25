import React, { useState, useEffect } from 'react';
import add2 from '../../../../images/add2.png';
import axios from 'axios';

function Update({ patientId }) {
  const apiUrl = process.env.REACT_APP_API_BASE_URL || "http://localhost:3000";
  const [selectedDoctor, setSelectedDoctor] = useState('');
  const [doctors, setDoctors] = useState([]);
  const [error, setError] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isClicked, setIsClicked] = useState(false);

  useEffect(() => {
    
    const fetchDoctors = async () => {
      try {
        const token = localStorage.getItem('token') || sessionStorage.getItem('token');
        if (!token) {
          throw new Error('No authentication token found');
        }

        const response = await axios.get(`${apiUrl}/doctors`, {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        });
        setDoctors(response.data);
      } catch (error) {
        console.error("Error fetching doctors:", error);
        setError(error.response?.data?.message || "Failed to load doctors");
      }
    };
    fetchDoctors();
  }, []);

  const handleAssignPatient = async () => {
    if (!patientId || !selectedDoctor) return alert("Please select both patient and doctor.");
    console.log("Assigning patient with:", { patientId, doctorId: selectedDoctor });

    try {
      const token = localStorage.getItem('token') || sessionStorage.getItem('token');
      if (!token) {
        throw new Error('No authentication token found');
      }

      await axios.post(`${apiUrl}/patients/assign-to-doctor`, 
        { patientId, doctorId: selectedDoctor },
        {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        }
      );
      alert("Patient assigned successfully!");
      closeModal();
    } catch (error) {
      console.error("Failed to assign patient:", error);
      alert(error.response?.data?.message || "Failed to assign patient.");
    }
  };

  const closeModal = () => {
    setIsModalOpen(false);
    setIsClicked(false);
    setSelectedDoctor('');
  };

  return (
    <div>
      <img
        src={add2}
        alt="assign"
        className={`w-12 h-12 cursor-pointer mt-2 rounded-full transition-all ${
          isClicked ? 'bg-[#00FF47] bg-opacity-30' : ''
        }`}
        onClick={() => {
          setIsModalOpen(true);
          setIsClicked(true);
        }}
      />

      {isModalOpen && (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
          <div className="bg-gray-800 text-white p-6 rounded-lg shadow-lg w-96">
            <h2 className="text-lg font-bold mb-4 text-center">Assign Doctor</h2>
            {error && <p className="text-red-400 text-center mb-2">{error}</p>}

            <select
              className="w-full p-3 rounded-md bg-gray-700 text-white focus:outline-none focus:ring-2 focus:ring-lime-500 mb-4"
              value={selectedDoctor}
              onChange={(e) => setSelectedDoctor(e.target.value)}
              disabled={doctors.length === 0}
            >
              <option value="" disabled>
                {doctors.length > 0 ? "Select Doctor" : "No doctors available"}
              </option>
              {doctors.map((doctor) => (
                <option key={doctor._id} value={doctor._id}>
                  Dr. {doctor.username} {doctor.lastName}
                </option>
              ))}
            </select>

            <div className="flex justify-between items-center">
              <button
                className="px-4 py-2 bg-gray-500 rounded-md hover:bg-gray-600"
                onClick={closeModal}
              >
                Cancel
              </button>
              <button
                className={`px-4 py-2 rounded-md ${
                  selectedDoctor
                    ? 'bg-lime-500 text-white hover:bg-lime-600'
                    : 'bg-gray-600 text-gray-400 cursor-not-allowed'
                }`}
                onClick={handleAssignPatient}
                disabled={!selectedDoctor}
              >
                + Assign
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Update;
