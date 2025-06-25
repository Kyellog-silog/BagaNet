import {useState, useEffect} from 'react'
import training from "../../images/blockchain.png"
import model from "../../images/model.png"
import weights from "../../images/weights.png"
import fone from "../../images/f1.png"
import accuracy from "../../images/accuracy.png"
import recall from "../../images/memory-recall.png"
import precision from "../../images/precision.png"
import history from "../../images/history.png"
import training2 from "../../images/work-in-progress.png"
import * as tf from '@tensorflow/tfjs';
import { Button } from "@/components/ui/button"
import {
  Drawer,
  DrawerClose,
  DrawerContent,
  DrawerDescription,
  DrawerFooter,
  DrawerHeader,
  DrawerTitle,
  DrawerTrigger,
} from "@/components/ui/drawer"
import {
  Carousel,
  CarouselContent,
  CarouselItem,
  CarouselNext,
  CarouselPrevious,
} from "@/components/ui/carousel"
// TensorFlow.js training utilities and submission
import { ds } from './data.js';
import { trainModel } from './training.js';
import { fetchTfjsModel, manualPurgeUsedImages } from './io.js';
import { getAllStoredImages, getStorageStats } from '../utils/indexedDBUtils.js';
import { prepareData } from './preprocessing.js'


const Metrics = () => {
  const [progress, setProgress] = useState(0)
  const [epochInfo, setEpochInfo] = useState([])
  const [error, setError] = useState(null)
  const [imageCount, setImageCount] = useState(0)
  const [isTraining, setIsTraining] = useState(false)
  const [purging, setPurging] = useState(false)
  const [localMetrics, setLocalMetrics] = useState({ date: '-', accuracy: '0', precision: '0', recall: '0', f1Score: '0' })
  const [metricsHistory, setMetricsHistory] = useState([])
  const [globalMetrics, setGlobalMetrics] = useState({ date: '-', accuracy: '0', precision: '0', recall: '0', f1Score: '0', version: '' })
  const [currentPhase, setCurrentPhase] = useState('training')
  const [weightsSendProgress, setWeightsSendProgress] = useState(0)

  // Function to determine background class based on current phase
  const getBgClass = (phase) =>
    currentPhase === phase ? 'bg-green-200 ring-green-400 font-bold' : 'bg-white';

  useEffect(() => {
    async function loadImages() {
      try {
        const stats = await getStorageStats();
        setImageCount(stats.totalImages || 0);
      } catch (err) {
        console.error("Error fetching image stats:", err);
        setError("Failed to fetch image stats");
      }
    }
    loadImages();
  
    fetchGlobalMetrics();
    
    // Set up interval to refresh global metrics every 30 seconds
    const intervalId = setInterval(fetchGlobalMetrics, 30000);
    
    // Clean up interval on unmount
    return () => clearInterval(intervalId);
  }, [])

  const handleTrainModel = async () => {
    setError(null);
    setIsTraining(true);
    setProgress(0);
    setEpochInfo([]);
  
    try {
      // Get all images from IndexedDB instead of using getImagesByTrainingStatus
      const images = await getAllStoredImages();
      
      if (images.length === 0) {
        setError("No images found in database. Please add images first.");
        setIsTraining(false);
        return;
      }
      
      // Update image count
      setImageCount(images.length);
      
      // Modified fetch model with better CORS handling
      const fetchModelWithRetry = async () => {
        try {
          return await fetchTfjsModel(p => setProgress(p * 0.2));
        } catch (fetchErr) {
          // If there's a CORS error, log it but don't fail immediately
          if (fetchErr.message && fetchErr.message.includes('CORS')) {
            console.error("CORS error detected, retrying with different approach:", fetchErr);
            
            // Simple retry with a slight delay
            await new Promise(resolve => setTimeout(resolve, 1000));
            
            // Attempt a different fetch approach 
            const modelRes = await fetch('http://localhost:5050/tfjs/training_model', {
              method: 'GET',
              headers: { 
                'X-API-Key': 'FeDMl2025', 
                'X-Client-ID': localStorage.getItem('clientId') 
              },
              mode: 'cors',
              credentials: 'same-origin'
            });
            
            if (!modelRes.ok) {
              throw new Error(`Failed to fetch model: ${modelRes.status}`);
            }
            
            const modelData = await modelRes.json();
            // Process the data similar to what fetchTfjsModel would do
            return {
              model: await tf.loadLayersModel(modelData.model_url),
              trainingParams: modelData.training_params
            };
          }
          throw fetchErr; // Re-throw if it's not a CORS error
        }
      };
      
      const { model, trainingParams } = await fetchModelWithRetry();
      setProgress(20);
  

      setCurrentPhase('validation');
      setProgress(25);


      setCurrentPhase('preprocessing');
      const prepared = await prepareData(images, p => setProgress(25 + p * 0.25));
  
      setCurrentPhase('testing');
      const results = await trainModel(
        prepared,                                // preparedData
        p      => setProgress(50 + p * 0.4),     // progressCallback
        epoch  => setEpochInfo(prev => [...prev, epoch]), // epochCallback
        true,                                    // submitToServer
        phase  => setCurrentPhase(phase),        // phaseCallback
        p      => setWeightsSendProgress(p)      // weightsProgressCallback
      );
  
      // Update local metrics
      setLocalMetrics({ 
        date: new Date().toLocaleDateString(), 
        accuracy: results.accuracy.toFixed(2), 
        precision: results.precision.toFixed(2), 
        recall: results.recall.toFixed(2), 
        f1Score: results.f1Score.toFixed(2) 
      });
      setMetricsHistory(prev => [results, ...prev]);

      console.log('Metrics.js Competed');
  
    } catch (err) {
      console.error("Training error:", err);
      setError(err.message || "Unknown training error occurred");
    } finally {
      setIsTraining(false);
      setProgress(0);
    }
  }

  const handlePurgeUsedImages = async () => {
    setPurging(true);
    try {
      const purgedCount = await manualPurgeUsedImages();
      
      // Update image stats after purging
      const stats = await getStorageStats();
      setImageCount(stats.totalImages || 0);
    } catch (err) {
      console.error("Error purging images:", err);
      setError("Failed to purge used images");
    } finally {
      setPurging(false);
    }
  }
  // Updated fetchGlobalMetrics function with better error handling
const fetchGlobalMetrics = async () => {
  try {
    const res = await fetch('http://localhost:5050/tfjs/metrics', {
      method: 'GET',
      headers: { 
        'X-API-Key': 'FeDMl2025', 
        'X-Client-ID': localStorage.getItem('clientId') 
      },
      // Add mode: 'cors' to explicitly request CORS
      mode: 'cors',
      // Add credentials handling if necessary
      credentials: 'same-origin'
    })
    
    if (!res.ok) {
      throw new Error(`Server responded with status: ${res.status}`);
    }
    
    const data = await res.json()
    
    if (data && data.metrics) {
      setGlobalMetrics({ 
        date: data.timestamp || new Date().toLocaleDateString(), 
        accuracy: data.metrics.accuracy || '-', 
        precision: data.metrics.precision || '-', 
        recall: data.metrics.recall || '-', 
        f1Score: data.metrics.f1_score || '-', 
        version: data.version || ''
      })
    }
  } catch (err) {
    console.error("Error fetching global metrics:", err);
    setError(`Failed to fetch global metrics: ${err.message}`);
    
    // Keep previous metrics if available
    if (globalMetrics.date === '-') {
      setGlobalMetrics({
        date: new Date().toLocaleDateString(),
        accuracy: 'N/A', 
        precision: 'N/A', 
        recall: 'N/A', 
        f1Score: 'N/A',
        version: 'unavailable'
      });
    }
  }
}


  return (
  
  <div className="max-w-screen-xl mx-auto grid grid-cols-1 xl:grid-cols-4 gap-8 bg-white rounded-2xl font-poppins p-10">
    
    <div className="col-span-2 row-span-1 rounded-xl space-y-6">
      <div className='ring-1 ring-gray-300 p-4 rounded-3xl bg-[#F4F6F6] shadow-lg'>
      <div className='flex flex-row items-center gap-2'>
            <img src={training} alt="" className='h-8 w-8'></img>
        <div className="text-black text-base font-normal">Progress</div>
        </div>
      
        <progress
        className="progress progress-success w-full"
        value={progress}
        max="100"
      ></progress>

      <div className="flex flex-row justify-between gap-2 mt-2">
        <div
          className={`ring-1 text-black items-center w-full flex flex-col justify-center h-10 text-xs px-4 rounded-xl ${getBgClass(
            'training'
          )}`}
        >
          Training
        </div>
        <div
          className={`ring-1 text-black items-center w-full flex flex-col justify-center h-10 text-xs px-4 rounded-xl ${getBgClass(
            'validation'
          )}`}
        >
          Validation
        </div>
        <div
          className={`ring-1 text-black items-center w-full flex flex-col justify-center h-10 text-xs px-4 rounded-xl ${getBgClass(
            'testing'
          )}`}
        >
          Testing
        </div>
      </div>
      </div>
        

  
  <div className='bg-[#F4F6F6] ring-1 ring-gray-300 text-black p-4 space-y-4 rounded-3xl relative shadow-lg'>
  <div className='flex flex-row items-center gap-2'>
            <img src={training2} alt="" className='h-8 w-8'></img>
        <div className="text-black text-base font-normal">Training Epochs</div></div>
  <div className='2xl:max-h-[500px] lg:max-h-[300px] max-h-[200px] overflow-auto'>
      <ul className="text-white space-y-2 text-sm p-2">
        {epochInfo.length > 0 ? (
          epochInfo.map((epoch, index) => (
            <li key={index} className='bg-white ring-1 ring-gray-300 text-black px-4 py-2 rounded-xl'>
              Epoch {epoch.epoch || index + 1}: 
              Loss: {epoch.trainLoss?.toFixed(4) || epoch.loss?.toFixed(4) || "-"}, 
              Accuracy: {epoch.trainAccuracy?.toFixed(4) || epoch.accuracy?.toFixed(4) || "-"},
              Val Loss: {epoch.valLoss?.toFixed(4) || "-"}, 
              Val Acc: {epoch.valAccuracy?.toFixed(4) || "-"}
            </li>
          ))
        ) : (
          // Default epochs display when no training data
          <>
            <li className='bg-white ring-1 ring-gray-300 text-black px-4 py-2 rounded-xl'>Epoch 1: Waiting for training to begin...</li>
            <li className='bg-white ring-1 ring-gray-300 text-black px-4 py-2 rounded-xl'>Epoch 2:</li>
            <li className='bg-white ring-1 ring-gray-300 text-black px-4 py-2 rounded-xl'>Epoch 3:</li>
            <li className='bg-white ring-1 ring-gray-300 text-black px-4 py-2 rounded-xl'>Epoch 4:</li>
            <li className='bg-white ring-1 ring-gray-300 text-black px-4 py-2 rounded-xl'>Epoch 5:</li>
            <li className='bg-white ring-1 ring-gray-300 text-black px-4 py-2 rounded-xl'>Epoch 6:</li>
            <li className='bg-white ring-1 ring-gray-300 text-black px-4 py-2 rounded-xl'>Epoch 7:</li>
            <li className='bg-white ring-1 ring-gray-300 text-black px-4 py-2 rounded-xl'>Epoch 8:</li>
          </>
        )}
      </ul>
      </div>
      <div>
        <div className='flex flex-row gap-2 items-center'>
          <img src={weights} alt="" className='h-6 w-6'></img>
        <p className="text-black text-sm">Sending Weights</p>
        </div>
       
        <progress
        className="progress progress-success w-full"
        value={weightsSendProgress}
        max="100"></progress>
      </div>

  </div>
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative">
          <strong className="font-bold">Error: </strong>
          <span className="block sm:inline">{error}</span>
        </div>
      )}
      <div className="flex flex-row items-center justify-between rounded-lg bg-gray-100 p-2 mb-2">
        <span className="text-sm font-medium text-gray-600">Available Images: {imageCount}</span>
        <span className="text-xs text-gray-500">{isTraining ? "Training in progress..." : "Ready for training"}</span>
      </div>
      <button 
    className="btn btn-block bg-success text-white hover:bg-emerald-400/90 shadow-none border-none rounded-lg font-medium"
    onClick={handleTrainModel}
    disabled={isTraining}
  >
    {isTraining ? 'Training...' : 'Train'}
  </button>
  <button
    className="btn btn-block bg-red-500 text-white hover:bg-red-400/90 shadow-none border-none rounded-lg font-medium"
    onClick={handlePurgeUsedImages}
    disabled={purging || isTraining}
  >
    {purging ? 'Purging...' : 'Purge Used'}
  </button>
    </div>
  


<div className='col-span-2 space-y-6'>


    <div className="rounded-xl p-6 flex flex-col gap-4 bg-[#F4F6F6] ring-1 ring-gray-300 text-black shadow-lg">
      <div className="flex flex-row gap-2">
        <img src={model} alt="model" className="w-24 h-24 rounded-2xl bg-slate-200 p-2" />
        <div className='flex flex-col w-full gap-1'> 
        <div className="flex flex-row items-center bg-[#F4F6F6] p-2">
  <div className="flex items-center gap-2 px-6 py-1 rounded-full text-xs font-light ring-1 ring-gray-300">
    <span className="h-2 w-2 rounded-full bg-green-600 inline-block ring-4 ring-green-200/90"></span>
    <span>Latest Model</span>
  </div>
</div>


          <h2 className="text-xl font-bold">Current Local Model Metrics</h2>
          <p className="text-sm text-black">Date: {localMetrics.date}</p>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-2 gap-4 mt-2">
        <div className="ring-1 ring-gray-300 flex flex-row justify-center rounded-md p-3">
          <div className='flex flex-row items-center gap-2'>
          <img src={accuracy} alt="" className='h-6 w-6'></img>
          
          <div className=''>
          Accuracy: {localMetrics.accuracy}
          </div>
          </div>

        </div>
        <div className="ring-1 ring-gray-300 flex flex-row justify-center rounded-md p-3">
          <div className='flex flex-row items-center gap-2'>
          <img src={precision} alt="" className='h-6 w-6'></img>
          
          <div>
          Precision: {localMetrics.precision}
          </div>
          </div>

        </div>
        <div className="ring-1 ring-gray-300 flex flex-row justify-center rounded-md p-3">
          <div className='flex flex-row items-center gap-2'>
          <img src={recall} alt="" className='h-6 w-6'></img>
          
          <div>
          Recall: {localMetrics.recall}
          </div>
          </div>

        </div>
        <div className="ring-1 ring-gray-300 flex flex-row justify-center rounded-md p-3">
          <div className='flex flex-row items-center gap-2'>
          <img src={fone} alt="" className='h-6 w-6'></img>
          
          <div>
          F1 Score: {localMetrics.f1Score}
          </div>
          </div>

        </div>
      </div>

      <Drawer>
      <DrawerTrigger>
      <div className="flex flex-row justify-end items-center gap-2 cursor-pointer">
        <img src={history} alt="" className='h-4 w-4'></img>
        <h1 className='text-sm font-extralight hover:text-slate-600'>History</h1>
      </div>
      
      </DrawerTrigger>
      <DrawerContent>
        <Carousel>
          <CarouselContent>
            {metricsHistory.length > 0 ? (
              metricsHistory.map((metrics, index) => (
                <CarouselItem key={metrics.id || index}>
                  <DrawerHeader>
                    <div className='flex flex-row justify-center items-center w-full'>
                      <div className="w-1/2 rounded-xl p-6 flex flex-col gap-4 bg-[#F4F6F6] ring-1 ring-gray-300 text-black shadow-lg">
                        
                        <div className="flex flex-row gap-2">
                          <img src={model} alt="model" className="w-24 h-24 rounded-2xl bg-slate-800 p-2" />
                          <div className='flex flex-col w-full gap-1'> 
                            <div className="flex flex-row items-center bg-[#F4F6F6] p-2">
                              <div className="flex items-center gap-2 px-6 py-1 rounded-full text-xs font-light ring-1 ring-gray-300">
                                <span className="h-2 w-2 rounded-full bg-green-600 inline-block ring-4 ring-green-200/90"></span>
                                <span>{index === 0 ? 'Latest Model' : `Model ${metricsHistory.length - index}`}</span>
                              </div>
                            </div>
                            <h2 className="text-xl font-bold">Local Model Metrics</h2>
                            <p className="text-sm text-black">Date: {metrics.date}</p>
                          </div>
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                          <div className="ring-1 ring-gray-300 flex flex-row justify-center rounded-md p-4">
                            <div className='flex flex-row items-center gap-2'>
                              <img src={accuracy} alt="" className='h-6 w-6' />
                              <div>Accuracy: {metrics.accuracy}</div>
                            </div>
                          </div>
                          <div className="ring-1 ring-gray-300 flex flex-row justify-center rounded-md p-4">
                            <div className='flex flex-row items-center gap-2'>
                              <img src={precision} alt="" className='h-6 w-6' />
                              <div>Precision: {metrics.precision}</div>
                            </div>
                          </div>
                          <div className="ring-1 ring-gray-300 flex flex-row justify-center rounded-md p-4">
                            <div className='flex flex-row items-center gap-2'>
                              <img src={recall} alt="" className='h-6 w-6' />
                              <div>Recall: {metrics.recall}</div>
                            </div>
                          </div>
                          <div className="ring-1 ring-gray-300 flex flex-row justify-center rounded-md p-4">
                            <div className='flex flex-row items-center gap-2'>
                              <img src={fone} alt="" className='h-6 w-6' />
                              <div>F1 Score: {metrics.f1Score}</div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </DrawerHeader>
                </CarouselItem>
              ))
            ) : (
              <CarouselItem>
                <DrawerHeader>
                  <div className='flex flex-row justify-center items-center w-full'>
                    <div className="w-1/2 rounded-xl p-6 flex flex-col gap-4 bg-[#F4F6F6] ring-1 ring-gray-300 text-black shadow-lg">
                      
                      <div className="flex flex-row gap-2">
                        <img src={model} alt="model" className="w-24 h-24 rounded-2xl bg-slate-800 p-2" />
                        <div className='flex flex-col w-full gap-1'> 
                          <div className="flex flex-row items-center bg-[#F4F6F6] p-2">
                            <div className="flex items-center gap-2 px-6 py-1 rounded-full text-xs font-light ring-1 ring-gray-300">
                              <span className="h-2 w-2 rounded-full bg-green-600 inline-block ring-4 ring-green-200/90"></span>
                              <span>Latest Model</span>
                            </div>
                          </div>
                          <h2 className="text-xl font-bold">Current Local Model Metrics</h2>
                          <p className="text-sm text-black">Date: {new Date().toLocaleDateString()}</p>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-4">
                        <div className="ring-1 ring-gray-300 flex flex-row justify-center rounded-md p-4">
                          <div className='flex flex-row items-center gap-2'>
                            <img src={accuracy} alt="" className='h-6 w-6' />
                            <div>Accuracy: - </div>
                          </div>
                        </div>
                        <div className="ring-1 ring-gray-300 flex flex-row justify-center rounded-md p-4">
                          <div className='flex flex-row items-center gap-2'>
                            <img src={precision} alt="" className='h-6 w-6' />
                            <div>Precision: - </div>
                          </div>
                        </div>
                        <div className="ring-1 ring-gray-300 flex flex-row justify-center rounded-md p-4">
                          <div className='flex flex-row items-center gap-2'>
                            <img src={recall} alt="" className='h-6 w-6' />
                            <div>Recall: - </div>
                          </div>
                        </div>
                        <div className="ring-1 ring-gray-300 flex flex-row justify-center rounded-md p-4">
                          <div className='flex flex-row items-center gap-2'>
                            <img src={fone} alt="" className='h-6 w-6' />
                            <div>F1 Score: - </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </DrawerHeader>
              </CarouselItem>
            )}
          </CarouselContent>

          <CarouselPrevious className="bg-white border rounded-full shadow p-2 absolute left-72 top-1/2 -translate-y-1/2 z-10" />
          <CarouselNext className="bg-white border rounded-full shadow p-2 absolute right-72 top-1/2 -translate-y-1/2 z-10" />
        </Carousel>
      </DrawerContent>
    </Drawer>
    </div>

    
    <div className="rounded-xl p-6 flex flex-col gap-4 bg-[#F4F6F6] ring-1 ring-gray-300 text-black shadow-lg">
      <div className="flex flex-row gap-2">
        <img src={training} alt="model" className="w-24 h-24 rounded-2xl bg-slate-200 p-2" />
        <div className='flex flex-col w-full gap-1'> 
        <div className="flex flex-row items-center bg-[#F4F6F6] p-2">
  <div className="flex items-center gap-2 px-6 py-1 rounded-full text-xs font-light ring-1 ring-gray-300">
    <span className="h-2 w-2 rounded-full bg-green-600 inline-block ring-4 ring-green-200/90"></span>
    <span>Latest Model {globalMetrics.version ? `v${globalMetrics.version}` : ''}</span>
  </div>
</div>


          <h2 className="text-xl font-bold">Current Global Model Metrics</h2>
          <p className="text-sm text-black">Date: {globalMetrics.date}</p>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-2 gap-4 mt-2">
        <div className="ring-1 ring-gray-300 flex flex-row justify-center rounded-md p-3">
          <div className='flex flex-row items-center gap-2'>
          <img src={accuracy} alt="" className='h-6 w-6'></img>
          
          <div className=''>
          Accuracy: {globalMetrics.accuracy}
          </div>
          </div>

        </div>
        <div className="ring-1 ring-gray-300 flex flex-row justify-center rounded-md p-3">
          <div className='flex flex-row items-center gap-2'>
          <img src={precision} alt="" className='h-6 w-6'></img>
          
          <div>
          Precision: {globalMetrics.precision}
          </div>
          </div>

        </div>
        <div className="ring-1 ring-gray-300 flex flex-row justify-center rounded-md p-3">
          <div className='flex flex-row items-center gap-2'>
          <img src={recall} alt="" className='h-6 w-6'></img>
          
          <div>
          Recall: {globalMetrics.recall}
          </div>
          </div>

        </div>
        <div className="ring-1 ring-gray-300 flex flex-row justify-center rounded-md p-3">
          <div className='flex flex-row items-center gap-2'>
          <img src={fone} alt="" className='h-6 w-6'></img>
          
          <div>
          F1 Score: {globalMetrics.f1Score}
          </div>
          </div>

        </div>
      </div>

      <div className="flex flex-row justify-end items-center">
        <button
          onClick={fetchGlobalMetrics}
          className="text-sm font-light hover:text-slate-600 flex items-center gap-1"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          Refresh
        </button>
      </div>

      <Drawer>
        <DrawerTrigger>
          <div className="flex flex-row justify-end items-center gap-2 cursor-pointer">
            <img src={history} alt="" className='h-4 w-4'></img>
            <h1 className='text-sm font-extralight hover:text-slate-600'>History</h1>
          </div>
          </DrawerTrigger>
  <DrawerContent>
  <Carousel>
    <CarouselContent className="">

      <CarouselItem>
        <DrawerHeader>
          <div className='flex flex-row justify-center items-center w-full '>
            <div className="w-1/2 rounded-xl p-6 flex flex-col gap-4 bg-[#F4F6F6] ring-1 ring-gray-300 text-black shadow-lg">
              
              <div className="flex flex-row gap-2">
                <img src={model} alt="model" className="w-24 h-24 rounded-2xl bg-slate-800 p-2" />
                <div className='flex flex-col w-full gap-1'> 
                  <div className="flex flex-row items-center bg-[#F4F6F6] p-2">
                    <div className="flex items-center gap-2 px-6 py-1 rounded-full text-xs font-light ring-1 ring-gray-300">
                      <span className="h-2 w-2 rounded-full bg-green-600 inline-block ring-4 ring-green-200/90"></span>
                      <span>Latest Model</span>
                    </div>
                  </div>
                  <h2 className="text-xl font-bold">Current Global Model Metrics</h2>
                  <p className="text-sm text-black">Date: {globalMetrics.date}</p>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="ring-1 ring-gray-300 flex flex-row justify-center rounded-md p-4">
                  <div className='flex flex-row items-center gap-2'>
                    <img src={accuracy} alt="" className='h-6 w-6' />
                    <div>Accuracy: {globalMetrics.accuracy}</div>
                  </div>
                </div>
                <div className="ring-1 ring-gray-300 flex flex-row justify-center rounded-md p-4">
                  <div className='flex flex-row items-center gap-2'>
                    <img src={precision} alt="" className='h-6 w-6' />
                    <div>Precision: {globalMetrics.precision}</div>
                  </div>
                </div>
                <div className="ring-1 ring-gray-300 flex flex-row justify-center rounded-md p-4">
                  <div className='flex flex-row items-center gap-2'>
                    <img src={recall} alt="" className='h-6 w-6' />
                    <div>Recall: {globalMetrics.recall}</div>
                  </div>
                </div>
                <div className="ring-1 ring-gray-300 flex flex-row justify-center rounded-md p-4">
                  <div className='flex flex-row items-center gap-2'>
                    <img src={fone} alt="" className='h-6 w-6' />
                    <div>F1 Score: {globalMetrics.f1Score}</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </DrawerHeader>
      </CarouselItem>

    </CarouselContent>

    <CarouselPrevious className="bg-white border rounded-full shadow p-2 absolute left-72 top-1/2 -translate-y-1/2 z-10" />
<CarouselNext className="bg-white border rounded-full shadow p-2 absolute right-72 top-1/2 -translate-y-1/2 z-10" />

  </Carousel>
</DrawerContent>

</Drawer>
    </div>
    </div>




  </div>
  )
}

export default Metrics