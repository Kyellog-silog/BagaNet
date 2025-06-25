import React from 'react'
import { useNavigate } from 'react-router-dom';
import lungss from '../../images/lungss.png';


const Topcontent = () => {
  const navigate = useNavigate();
  return (
    <div className='mt-2'>
    
       <section className="container mx-auto flex justify-between items-center py-10 px-6  mb-14 mt-10">
        <div className="container mx-auto flex flex-col md:flex-row items-center">
          <div className="md:w-3/6 ">
          <img src={lungss} alt="lungss" className=" object-contain rounded-lg shadow-sm animate-floating" /> 
          </div>
          <div className="md:w-1/2 md:pl-12 mt-8 md:mt-0 text-center md:text-left">
            <h1 className="xl:text-2xl md:text-xl text-lg font-bold text-slate-600">
              Welcome to <span className=''>BAGA.NET</span>
            </h1>
            <h2 className="xl:text-5xl md:text-3xl text-2xl font-bold text-slate-900 mt-4 ">
              Revolutionizing Lung Disease Classification with FedML
            </h2>
            <p className="text-slate-700 mt-4 xl:text-base text-xs">
              Streamline your workflow and enhance diagnostic accuracy with our cutting-edge federated machine
              learning platform, designed for doctors and radiologic technologists.
            </p>
            <div className="relative inline-flex mt-6">

            <button
              className="relative border-2 border-teal-500 text-black  px-6 py-3 rounded-lg"
              onClick={() => navigate('/models')}
            >
              Discover More
            </button>

            <span className="absolute top-1 right-1 transform translate-x-1/2 -translate-y-1/2">
              <span className="animate-ping absolute inline-flex h-4 w-4 rounded-full bg-teal-600 opacity-50"></span>
              <span className="relative inline-flex rounded-full h-4 w-4 bg-teal-500"></span>
            </span>
          </div>
            
          </div>
        </div>
      </section>
    </div>
  )
}

export default Topcontent
