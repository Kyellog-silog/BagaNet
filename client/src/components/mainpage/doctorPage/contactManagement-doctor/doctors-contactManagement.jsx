import React from 'react'
import DoctorsNavbar from '../dasboard-doctor/doctors-navbar';
import DoctorsSidebar from '../dasboard-doctor/doctors-sidebar';


function DoctorscontactManagement() {
  return (
    <div className="flex h-screen bg-gray-100">
    <DoctorsSidebar/>
    <div className="flex flex-col flex-grow">
    <DoctorsNavbar />
    </div>
    </div>
  )
}

export default DoctorscontactManagement
