import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { FaEye, FaEyeSlash, FaSpinner } from 'react-icons/fa';
import { AiOutlineMail } from 'react-icons/ai';
import { LuLock } from 'react-icons/lu';
import { FcGoogle } from 'react-icons/fc';
import { FaApple } from 'react-icons/fa';

function Login() {
  const apiUrl = process.env.REACT_APP_API_BASE_URL || "http://localhost:3000";
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [rememberMe, setRememberMe] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    try {
      const response = await axios.post(`${apiUrl}/patients/login`, { 
        email, 
        password 
      });

      if (response.data.message === 'Success') {
        // Store token securely
        const token = response.data.token;
        if (rememberMe) {
          localStorage.setItem('token', token);
        } else {
          sessionStorage.setItem('token', token);
        }

        // Store user data
        localStorage.setItem('username', response.data.username || "Guest");
        localStorage.setItem('userId', response.data._id);
        localStorage.setItem('userRole', response.data.role);

        if (response.data.role === 'doctor') {
          localStorage.setItem('doctorId', response.data.doctorId);
          navigate('/doctorsPage');
        } else if (response.data.role === 'radtech') {
          navigate('/radtechsPage');
        } else {
          navigate('/');
        }
      }
    } catch (error) {
      // Handle error without exposing sensitive information
      setError('Login failed. Please check your credentials.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex justify-center items-center min-h-screen bg-gray-100">
      <div className="relative p-8 rounded-lg shadow-lg max-w-sm w-full bg-white">
        <div className="absolute top-0 opacity-30 rounded-t-lg left-0 w-full h-48 bg-gradient-to-b from-blue-500 to-transparent"></div>

        <h2 className="text-2xl font-semibold text-slate-900 mb-6 text-center">Welcome Back</h2>
        {error && (
          <div className="mb-4 p-3 bg-red-100 text-red-700 rounded-lg text-sm">
            {error}
          </div>
        )}
        <form onSubmit={handleSubmit}>
          {/* Email Input */}
          <div className="mb-4 relative">
            <label className="block text-gray-700 font-medium mb-2" htmlFor="email">Email</label>
            <div className="relative">
              <span className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-500 text-lg">
                <AiOutlineMail />
              </span>
              <input
                type="email"
                id="email"
                placeholder="Enter your email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full pl-10 pr-3 py-2 border rounded-lg focus:outline-none focus:ring focus:border-blue-300"
                required
              />
            </div>
          </div>

          {/* Password Input */}
          <div className="mb-6 relative">
            <label className="block text-gray-700 font-medium mb-2" htmlFor="password">Password</label>
            <div className="relative">
              <span className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-500 text-lg">
                <LuLock />
              </span>
              <input
                type={showPassword ? 'text' : 'password'}
                id="password"
                placeholder="Enter your password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full pl-10 pr-3 py-2 border rounded-lg focus:outline-none focus:ring focus:border-blue-300"
                required
              />
              <span 
                className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-500 cursor-pointer" 
                onClick={() => setShowPassword(!showPassword)}
              >
                {showPassword ? <FaEyeSlash /> : <FaEye />}
              </span>
            </div>
          </div>

          {/* Remember Me Checkbox */}
          <div className="mb-6 flex items-center">
            <input
              type="checkbox"
              id="rememberMe"
              checked={rememberMe}
              onChange={() => setRememberMe(!rememberMe)}
              className="mr-2"
            />
            <label htmlFor="rememberMe" className="text-gray-700">Remember me</label>
          </div>

          {/* Submit Button */}
          <button 
            type="submit" 
            className="w-full bg-blue-500 text-white py-2 rounded-lg hover:bg-blue-600 transition-colors flex items-center justify-center"
            disabled={isLoading}
          >
            {isLoading ? <FaSpinner className="animate-spin mr-2" /> : 'Login'}
          </button>

          {/* Additional Options */}
          <div className="flex flex-col mt-4 justify-center items-center">
            <div className="flex flex-row mt-4 justify-center">
              <div>Don't have an account? <span className="text-blue-600 cursor-pointer">Sign Up</span></div>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}

export default Login;
