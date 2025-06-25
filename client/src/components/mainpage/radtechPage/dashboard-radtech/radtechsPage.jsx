import React from 'react';
import Navbar from './navbar';
import PatientActivity from '../patientRecords-radtech/patientActivity';
import Sidebar from './sidebar';
import Charts from './charts';
import dashPatient from '../../../../images/dashPatient.png'
import doctors from '../../../../images/doctors.png';
import num from '../../../../images/num.png';
import redArrow from '../../../../images/redArrow.png';
import greenArrow from '../../../../images/greenArrow.png';
import Dashboard from '../dashboard-radtech/dashboard';

function RadtechsPage() {
  return (
   <div>
    <Dashboard/>
   </div>
  );
}

export default RadtechsPage;
