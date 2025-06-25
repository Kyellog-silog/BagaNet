import {useEffect, useState} from 'react';
import axios from "axios";
import Navbar from './doctors-navbar';
import Sidebar from './doctors-sidebar';
import Charts from './charts';
import dashPatient from '../../../../images/dashPatient.png'
import doctors from '../../../../images/doctors.png';

function Doctorsdashboard() {
  const apiUrl = process.env.REACT_APP_API_BASE_URL || "http://localhost:3000";
  
  const [counts, setCounts] = useState({ totalDoctors: 0, totalRadtechs: 0, totalPatients: 0 });
  
  useEffect(() => {
    const token = localStorage.getItem('token') || sessionStorage.getItem('token');
    if (!token) {
      console.error('No authentication token found');
      return;
    }

    axios.get(`${apiUrl}/patients/dashboard-counts`, {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    })
      .then((response) => {
        setCounts(response.data);
      })
      .catch((error) => {
        console.error("Error fetching dashboard counts:", error);
      });
  }, []);

  return (
    <div className="flex h-screen bg-gray-100">
      <Sidebar />
      <div className="flex flex-col flex-grow">
        <Navbar />
        <div className="p-6 flex flex-col">
          <div className='text-4xl font-bold'>Dashboard </div>
          <div className='mt-2'>A quick data overview of the clinic.</div>
        </div> 
        <div className="p-6 -mt-2 flex flex-col space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-white p-4 rounded shadow-md">
              <img src={doctors} alt=""></img>
              <h3 className="text-lg font-semibold mt-2 mb-2">Total Doctors</h3>
              <p className="text-2xl font-bold text-red-600">{counts.totalDoctors}</p>
            </div>
            <div className="bg-white p-4 rounded shadow-md">
              <img src={doctors} alt=""></img>
              <h3 className="text-lg font-semibold mt-2 mb-2">Total Radtechs</h3>
              <p className="text-2xl font-bold text-red-600">{counts.totalRadtechs}</p>
            </div>
            <div className="bg-white p-4 rounded shadow-md">
              <img src={dashPatient} alt=""></img>
              <h3 className="text-lg font-semibold mt-2 mb-2">Current Total Patients</h3>
              <p className="text-2xl font-bold text-red-600">{counts.totalPatients}</p>
            </div>
          </div>

          <Charts />
        </div>
      </div>
    </div>
  )
}

export default Doctorsdashboard
