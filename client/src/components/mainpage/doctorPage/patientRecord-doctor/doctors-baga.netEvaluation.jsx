import React from 'react'
import xray_result from "../../../../images/xray_picture.png"

export default function BAGANETEvaluation() {
  return (
    <div>
        <div className='bg-white flex flex-row  justify-between p-6 rounded-lg h-[30vh] '>
          <div className='w-full flex flex-col text-black '>
            <div className='text-2xl  font-bold'>
                Classified Disease:
            </div>
            <dzv className='text-base '>
                Classified Disease 
            </dzv>
          </div>    
          <div className='w-1/6 flex flex-row justify-center items-end text-white   '>
          <button
                className="px-4 py-2 bg-blue-600 rounded-md hover:bg-blue-900"
              >
                Generate
              </button>
          </div>    
        </div>
      
    </div>
  )
}
