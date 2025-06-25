import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Spin as Hamburger } from "hamburger-react"; 
import { motion } from "framer-motion"; 
import {
  FaHome,
  FaInfoCircle,
  FaProjectDiagram,
  FaEnvelope,
} from "react-icons/fa"; 
import bagalogo from "../../images/bagalogo.png";

const Hero = () => {
  const [isSidebarOpen, setSidebarOpen] = useState(false); 
  const navigate = useNavigate();

  return (
    <nav className="xl:mt-10">
      <div className="container mx-auto flex justify-between items-center py-4 px-8  shadow-lg rounded-sm">
        <div className="flex items-center">
          <img src={bagalogo} alt="Baga" className="xl:h-5 h-10 object-contain  hidden sm:block" />
          <span className="text-xl font-bold ml-2 hidden sm:block ">BAGA.NET</span>
        </div>

        <div className="flex items-center space-x-8">
          <div className="lg:hidden ">
            <Hamburger toggled={isSidebarOpen} toggle={setSidebarOpen} />
          </div>
          <div className="hidden lg:flex space-x-14">
            <a href="#" className="text-gray-700 hover:text-blue-600">
              Home
            </a>
            <a href="#" className="text-gray-700 hover:text-blue-600">
              About
            </a>
            <a href="#" className="text-gray-700 hover:text-blue-600">
              Models
            </a>
            <a href="#" className="text-gray-700 hover:text-blue-600">
              Contacts
            </a>
          </div>
        </div>

        <div className="space-x-4  lg:flex">
          <button
            className="text-gray-700 hover:text-blue-600"
            onClick={() => navigate("/login")}
          >
            Login
          </button>
          <button
            className="bg-teal-500 text-white px-4 py-2 rounded hover:bg-blue-700"
            onClick={() => navigate("/signup")}
          >
            Sign Up
          </button>
        </div>
      </div>

      {isSidebarOpen && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.9 }}
          className="absolute top-16 right-4 bg-[#2c2f3e] text-purple-200 rounded-lg shadow-lg py-4 px-6 w-48 z-50"
        >
          <div className="flex flex-col space-y-4">
            <div
              className="flex items-center space-x-2 cursor-pointer hover:text-purple-500"
              onClick={() => {
                setSidebarOpen(false);
                navigate("/");
              }}
            >
              <FaHome />
              <span>Home</span>
            </div>
            <div
              className="flex items-center space-x-2 cursor-pointer hover:text-purple-500"
              onClick={() => {
                setSidebarOpen(false);
                navigate("/about");
              }}
            >
              <FaInfoCircle />
              <span>About</span>
            </div>
            <div
              className="flex items-center space-x-2 cursor-pointer hover:text-purple-500"
              onClick={() => {
                setSidebarOpen(false);
                navigate("/models");
              }}
            >
              <FaProjectDiagram />
              <span>Projects</span>
            </div>
            <div
              className="flex items-center space-x-2 cursor-pointer hover:text-purple-500"
              onClick={() => {
                setSidebarOpen(false);
                navigate("/contact");
              }}
            >
              <FaEnvelope />
              <span>Contact</span>
            </div>
          </div>
        </motion.div>
      )}
    </nav>
  );
};

export default Hero;
