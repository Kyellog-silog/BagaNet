import React from 'react';
import location from '../../images/location.png';
import email from '../../images/email.png';
import call from '../../images/call.png';

const Navbar = () => {
  return (
    <div className="bg-gray-800 text-white text-sm xl:text-xs">
      <div className="container mx-auto py-2 px-8 flex items-center justify-between">
        <div className="flex items-center">
          <div className="pr-2 hidden sm:block">
            <img src={location} alt="Location" className="h-5 object-contain" />
          </div>
          <div className="hidden sm:block">
            #Alangilan, Batangas City
          </div>
        </div>

        
          <div className="flex items-center">
            <div className="pr-2">
              <img src={email} alt="Email" className="h-5 object-contain hidden sm:block" />
            </div>
            <a href="https://gmail.com/" className="hidden sm:inline hover:underline px-2">
              BAGA.NET@health.care
            </a>
          </div>

          <div className="flex items-center">
            <div className="pr-2">
              <img src={call} alt="Call" className="h-5 object-contain hidden sm:block" />
            </div>
            <a href="https://web.whatsapp.com/" className="hidden sm:inline hover:underline px-2">
              Connect on Whatsapp
            </a>
          </div>
        </div>
      </div>
    
  );
};

export default Navbar;
