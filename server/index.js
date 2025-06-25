require("dotenv").config();
const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const rateLimit = require('express-rate-limit');
const projectRoutes = require("./routes/projectRoutes");
const doctorRoutes = require("./routes/doctorRoutes");
const modelRoutes = require("./routes/modelRoutes"); 
const path = require('path');
const app = express();

app.set('trust proxy', 1);

app.use('/models', express.static(path.join(__dirname, 'static/models')));;

const limiter = rateLimit({
  windowMs: 60 * 1000, 
  max: 300, 
  standardHeaders: true,
  legacyHeaders: false, 
  skipFailedRequests: true, 
  keyGenerator: (req) => {
    // Use the real IP address from the request
    return req.ip || req.connection.remoteAddress;
  }
});

// Apply rate limiting to all routes
app.use(limiter);

// Stricter rate limit for auth routes
const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 10, // limit each IP to 10 requests per windowMs
  standardHeaders: true,
  legacyHeaders: false,
  skipFailedRequests: true,
  keyGenerator: (req) => {
    return req.ip || req.connection.remoteAddress;
  }
});
const corsOptions = {
  origin: [
    "http://localhost:5173", 
    "https://baga-net.vercel.app",
    "https://baga-net-backend.vercel.app", 
    "http://localhost:5050", 
    "http://localhost:3000", // Add this
    "https://2f58-158-62-8-230.ngrok-free.app"
  ],
  methods: ["GET", "POST", "DELETE", "PUT", "OPTIONS", "PATCH"], // Add PATCH if needed
  allowedHeaders: [
    "Content-Type", 
    "Authorization", 
    "ngrok-skip-browser-warning",
    "X-API-Key", // Add this if you're using API keys
    "Accept" // Add this
  ],
  credentials: true,
  optionsSuccessStatus: 200 // Some legacy browsers choke on 204
};
app.use(express.json({ limit: "100mb" }));
app.use(cors(corsOptions));

// Root route handler
app.get('/', (req, res) => {
  res.json({ 
    message: 'BAGA.NET API is running',
    status: 'ok',
    timestamp: new Date().toISOString()
  });
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy',
    timestamp: new Date().toISOString()
  });
});

// JWT Secret check
if (!process.env.JWT_SECRET) {
  console.error("JWT_SECRET is missing. Check your .env file.");
  process.exit(1);
}

const mongoURI = process.env.MONGO_URI;
if (!mongoURI) {
  console.error("MONGO_URI is missing. Check your .env file.");
  process.exit(1);
}

mongoose.connect(mongoURI)
  .then(() => console.log("MongoDB Connected"))
  .catch((err) => console.error("MongoDB Connection Error:", err));

// Apply auth rate limiting to auth routes
app.use('/patients/login', authLimiter);
app.use('/patients/signup', authLimiter);

// Routes
app.use('/patients', projectRoutes);
app.use('/doctors', doctorRoutes);
app.use('/models', modelRoutes); 

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ 
    message: 'Something went wrong!',
    error: process.env.NODE_ENV === 'development' ? err.message : undefined
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({ 
    message: 'Route not found',
    path: req.originalUrl
  });
});

const PORT = process.env.PORT || 3000;

// Only start the server if we're not in a Vercel serverless function
if (process.env.NODE_ENV !== 'production' || !process.env.VERCEL) {
  app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
  });
}

// Export for Vercel
module.exports = app;