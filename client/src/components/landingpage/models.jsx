import React from "react";
import bg from '../../images/bg.png'; 


const PretrainedModels = () => {
  return (
    <div className="relative mt-32 ">
      <img src={bg} alt="Background" className="absolute inset-0 h-full w-full object-cover" />
      <div className="absolute inset-0 bg-black opacity-50"></div>

      <div className="relative flex flex-row justify-center items-center mx-auto  w-5/6 ">
        <div className="  mt-16">
         
        <h1 className="flex flex-row justify-center items-center  xl:text-4xl md:text-3xl text-2xl text-center font-semibold text-slate-200">Empowering Diagnostics with Advanced AI Models</h1>
      

        <div className="text-white mt-12 grid xl:grid-cols-3 grid-cols-1 gap-4  p-6">

          <div className=" p-2 flex flex-cols-2 gap-2 transition-transform transform hover:scale-105 hover:bg-slate-700 hover:rounded-lg">
          <div className=" "><div className="bg-blue-500 rounded-full px-4 py-2 text-center mt-20">1</div></div>
          <div className=" w-full p-4">
          <div className="text-xl font-medium xl:text-center md:text-center text-left text-slate-200 ">Alex-Net</div>
          <div className="mt-4 text-left text-sm text-slate-400"> is a deep convolutional neural network (CNN) designed by Alex Krizhevsky for the ImageNet competition in 2012, which significantly improved image classification performance by utilizing deep layers, ReLU activations, and dropout regularization.</div>
          </div></div>


          <div className=" p-2 flex flex-cols-2 gap-2 transition-transform transform hover:scale-105 hover:bg-slate-700 hover:rounded-lg ">
          <div className=" "><div className="bg-blue-500 rounded-full px-4 py-2 text-center mt-20">2</div></div>
          <div className="  w-full p-4">
          <div className="text-xl font-medium xl:text-center md:text-center text-left text-slate-200 ">RestNet</div>
          <div className="mt-4 text-left text-sm text-slate-400">A deep CNN that introduced residual learning with skip connections, allowing extremely deep networks to be trained by mitigating vanishing gradients. It revolutionized deep learning by enabling models with hundreds of layers.</div>
          </div></div>

          <div className=" p-2 flex flex-cols-2 gap-2 transition-transform transform hover:scale-105 hover:bg-slate-700 hover:rounded-lg ">
          <div className=" "><div className="bg-blue-500 rounded-full px-4 py-2 text-center mt-20">3</div></div>
          <div className="  w-full p-4">
          <div className="text-xl font-medium xl:text-center md:text-center text-left text-slate-200 ">Inception v4</div>
          <div className="mt-4 text-left  text-sm text-slate-400"> An improved version of the Inception architecture that combines residual connections with the Inception modules, enhancing both accuracy and training efficiency for image classification tasks.</div>
          </div></div>


          <div className=" p-2 flex flex-cols-2 gap-2 transition-transform transform hover:scale-105 hover:bg-slate-700 hover:rounded-lg ">
          <div className=" "><div className="bg-blue-500 rounded-full px-4 py-2 text-center mt-20">4</div></div>
          <div className="   w-full p-4">
          <div className="text-xl font-medium xl:text-center md:text-center text-left text-slate-200 ">LungNet</div>
          <div className="mt-4 text-left text-sm text-slate-400"> A specialized deep learning model designed for detecting and classifying lung diseases from medical images, often leveraging CNN architectures for feature extraction and diagnosis.</div>
          </div></div>

          <div className=" p-2 flex flex-cols-2 gap-2 transition-transform transform hover:scale-105 hover:bg-slate-700 hover:rounded-lg ">
          <div className=" "><div className="bg-blue-500 rounded-full px-4 py-2 text-center mt-20">5</div></div>
          <div className="  w-full p-4">
          <div className="text-xl font-medium xl:text-center md:text-center text-left text-slate-200 ">VGG-19</div>
          <div className="mt-4 text-left text-sm text-slate-400"> A CNN with 19 layers, known for its simplicity and uniform architecture, using small 3×3 convolutional filters to achieve high performance in image recognition tasks.</div>
          </div></div>


          <div className=" p-2 flex flex-cols-2 gap-2 transition-transform transform hover:scale-105 hover:bg-slate-700 hover:rounded-lg ">
          <div className=" "><div className="bg-blue-500 rounded-full px-4 py-2 text-center mt-20">6</div></div>
          <div className="   w-full p-4">
          <div className="text-xl font-medium xl:text-center md:text-center text-left text-slate-200  ">DenseNet</div>
          <div className="mt-4 text-left text-sm text-slate-400"> A CNN where each layer is connected to every previous layer, improving feature reuse and gradient flow while reducing the number of parameters needed for deep networks.</div>
          </div></div>

          <div className=" p-2 flex flex-cols-2 gap-2 transition-transform transform hover:scale-105 hover:bg-slate-700 hover:rounded-lg ">
          <div className=" "><div className="bg-blue-500 rounded-full px-4 py-2 text-center mt-20">7</div></div>
          <div className="  w-full p-4">
          <div className="text-xl font-medium xl:text-center md:text-center text-left text-slate-200 ">MobileNet</div>
          <div className="mt-4 text-left text-sm text-slate-400"> A lightweight and efficient CNN architecture optimized for mobile and embedded devices, utilizing depthwise separable convolutions to reduce computational cost while maintaining accuracy.</div>
          </div></div>
         
        </div>
      </div>
      </div>


      
    </div>
  );
};


export default PretrainedModels;
