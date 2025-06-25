
import React, { useState } from 'react';

function ImageSlider({ images }) {
    const [currentIndex, setCurrentIndex] = useState(0);
    const [isEnlarged, setIsEnlarged] = useState(false);

    if (!images || images.length === 0) {
        return <p>No images available</p>;
    }

    const nextImage = () => {
        setCurrentIndex((prevIndex) => (prevIndex + 1) % images.length);
    };

    const prevImage = () => {
        setCurrentIndex((prevIndex) => (prevIndex - 1 + images.length) % images.length);
    };

    const handleEnlarge = () => {
        setIsEnlarged(true);
    };

    const closeEnlarged = () => {
        setIsEnlarged(false);
    };

    return (
        <div className="relative">
        <img 
            src={`data:image/jpeg;base64,${images[currentIndex]}`} 
            alt={`X-ray ${currentIndex + 1}`} 
            className="w-40 h-40 object-cover rounded-md cursor-pointer" 
            onClick={handleEnlarge}
        />
        {images.length > 1 && (
            <>
            <button 
                className="absolute left-0 top-1/2 transform -translate-y-1/2 bg-gray-700 text-white p-2 rounded-full"
                onClick={prevImage}
            >
                &#9664;
            </button>
            <button 
                className="absolute right-0 top-1/2 transform -translate-y-1/2 bg-gray-700 text-white p-2 rounded-full"
                onClick={nextImage}
            >
                &#9654;
            </button>
            </>
        )}
        {isEnlarged && (
            <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-80 z-50">
            <div className="relative">
                <img 
                src={`data:image/jpeg;base64,${images[currentIndex]}`} 
                alt={`X-ray ${currentIndex + 1}`} 
                className="max-w-full max-h-full object-contain"
                />
                <button 
                className="absolute top-4 right-4 bg-gray-700 text-white p-2 rounded-full"
                onClick={closeEnlarged}
                >
                ✕
                </button>
                {images.length > 1 && (
                <>
                    <button 
                    className="absolute left-4 top-1/2 transform -translate-y-1/2 bg-gray-700 text-white p-2 rounded-full"
                    onClick={prevImage}
                    >
                    &#9664;
                    </button>
                    <button 
                    className="absolute right-4 top-1/2 transform -translate-y-1/2 bg-gray-700 text-white p-2 rounded-full"
                    onClick={nextImage}
                    >
                    &#9654;
                    </button>
                </>
                )}
            </div>
            </div>
        )}
        </div>
    );
    }

export default ImageSlider;
