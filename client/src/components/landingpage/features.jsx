import React, { useState } from 'react';
import federated from '../../images/federated.png';
import advanced from '../../images/advanced.png';
import crossvalidation from '../../images/crossvalidation.png';
import dataprivacy from '../../images/dataprivacy.png';
import continuouslearning from '../../images/continuouslearning.png';

const FeaturesList = () => {
  const [showMore, setShowMore] = useState(false);

  return (
    <section>
      <div className="container mx-auto px-6">
        <div className='flex flex-col justify-center items-center p-4 mb-8 gap-4'>
        <h2 className="xl:text-4xl md:text-3xl text-2xl text-center font-semibold text-gray-800 ">Empowering Diagnosis Through Innovation</h2>
        <h4 className=' xl:w-3/5 w-full text-center text-slate-700 xl:text-base text-xs '>Discover a suite of innovative features that enhance diagnostic precision, secure data integrity, and foster continuous learning for improved healthcare outcomes.</h4> </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 p-4 mt-12">
          {/* Feature Card 1 */}
          <div className="bg-white p-6 rounded-lg shadow-md border-t-4 border-blue-500">
            <div className="flex items-center mb-4">
              <img src={federated} alt="Federated" className="h-12 w-12" />
              <h3 className="xl:text-xl text-base font-semibold text-gray-800 ml-4">Federated Machine Learning</h3>
            </div>
            <p className="text-gray-600 xl:text-sm text-xs">Train AI models on decentralized data, maintaining high accuracy while prioritizing patient confidentiality and data security.</p>
          </div>
          {/* Feature Card 2 */}
          <div className="bg-white p-6 rounded-lg shadow-md">
            <div className="flex items-center mb-4">
              <img src={dataprivacy} alt="Dataprivacy" className="h-12 w-12" />
              <h3 className=" font-semibold text-gray-800 ml-4 xl:text-xl text-base ">Data Privacy</h3>
            </div>
            <p className="text-gray-600 xl:text-sm text-xs ">Built with cutting-edge security protocols to keep your data safe, ensuring compliance with the highest standards of privacy protection.</p>
          </div>
          {/* Feature Card 3 */}
          <div className="bg-white p-6 rounded-lg shadow-md">
            <div className="flex items-center mb-4">
              <img src={advanced} alt="Advanced" className="h-12 w-14" />
              <h3 className=" font-semibold text-gray-800 ml-2 xl:text-xl text-base ">Advance Models</h3>
            </div>
            <p className="text-gray-600 xl:text-sm text-xs">Access the most sophisticated AI architectures, including EfficientNetV2, DenseNet-201, and ResNet-152, to deliver exceptional diagnostic insights.</p>
          </div>
          {/* Feature Card 4 */}
          <div className="bg-white p-6 rounded-lg shadow-md">
            <div className="flex items-center mb-4">
              <img src={crossvalidation} alt="Crossvalidation" className="h-12 w-14" />
              <h3 className="font-semibold text-gray-800 ml-2 xl:text-xl text-base ">Cross-Validation for Doctors</h3>
            </div>
            <p className="text-gray-600 xl:text-sm text-xs">Enable effective collaboration between healthcare professionals, allowing seamless verification and alignment for better decision-making.</p>
          </div>
        </div>

        {/* Expandable Section */}
        <div className={`transition-all duration-500 ease-in-out overflow-hidden ${showMore ? 'max-h-screen' : 'max-h-0'}`}>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-8 mb-2">
            {/* Last Feature Card */}
            <div className="bg-white  p-6 rounded-lg shadow-md">
              <div className="flex items-center mb-4">
                <img src={continuouslearning} alt="Continuouslearning" className="h-12 w-14" />
                <h3 className="xl:text-xl text-base font-semibold text-gray-800 ml-2">Continuous Learning</h3>
              </div>
              <p className="text-gray-600 xl:text-sm text-xs">Empower the system to adapt and evolve, integrating new data to refine and improve diagnostic accuracy over time.</p>
            </div>
          </div>
        </div>

        {/* Toggle Button */}
        <div className="text-right mt-8">
          <button
            onClick={() => setShowMore(!showMore)}
            className="bg-teal-500 text-white px-6 py-3 rounded-lg hover:bg-teal-600 transition duration-300"
          >
            {showMore ? 'Hide Features' : 'See all Features'}
          </button>
        </div>
      </div>
    </section>
  );
};

export default FeaturesList;
