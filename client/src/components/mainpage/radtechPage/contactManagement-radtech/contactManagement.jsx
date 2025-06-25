import React from 'react'
import Sidebar from '../dashboard-radtech/sidebar'
import Navbar from '../dashboard-radtech/navbar'

function contactManagement() {
  return (
    <div className="flex h-screen bg-gray-100">
    <Sidebar />
    <div className="flex flex-col flex-grow">
      <Navbar />
    </div>
    </div>
  )
}

export default contactManagement
