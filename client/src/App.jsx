import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';

import Login from './auth/login';
import Signup from './auth/signup';
import Navbar from './components/landingpage/navbar';
import Hero from './components/landingpage/hero';
import Topcontent from './components/landingpage/topcontent';
import FeaturesList from './components/landingpage/features';
import PreTrainedModels from './components/landingpage/models';     
import About from './components/landingpage/about';
import DoctorsPage from './components/mainpage/doctorPage/dasboard-doctor/doctorsPage';
import RadtechsPage from './components/mainpage/radtechPage/dashboard-radtech/radtechsPage';
import Dashboard from './components/mainpage/radtechPage/dashboard-radtech/dashboard';
import PatientActivity from './components/mainpage/radtechPage/patientRecords-radtech/patientActivity';
import Doctorsdashboard from './components/mainpage/doctorPage/dasboard-doctor/doctors-dashboard';
import ContactManagement from './components/mainpage/radtechPage/contactManagement-radtech/contactManagement';
import DoctorscontactManagement from './components/mainpage/doctorPage/contactManagement-doctor/doctors-contactManagement';
import DoctorsPatientActivity from './components/mainpage/doctorPage/patientRecord-doctor/doctors-patientActivity';
import Metrics from './components/adminPanel/metrics';


function Home() {
  return (
    <>
      <Navbar />
      <Hero />
      <Topcontent />
      <About />
      <FeaturesList />
      <PreTrainedModels />
    </>
  );
}

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />
        <Route path="/home" element={<Home />} />
        <Route path="/doctorsPage" element={<DoctorsPage />} />
        <Route path="/radtechsPage" element={<RadtechsPage />} />
        <Route path="/dashboard" element={<Dashboard/>}/>
        <Route path="/doctors-dashboard" element={<Doctorsdashboard/>}/>
        <Route path="/doctors-patientActivity" element={<DoctorsPatientActivity/>}/>
        <Route path="/patientActivity" element={<PatientActivity/>}/>
        <Route path="/contactManagement" element={<ContactManagement/>}/>
        <Route path="/doctors-contactManagement" element={<DoctorscontactManagement/>}/>
        <Route path="/metrics" element={<Metrics/>}/>
        
        
      </Routes>
    </Router>
  );
}

export default App;
