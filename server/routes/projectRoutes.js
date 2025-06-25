const mongoose = require("mongoose");
const express = require("express");
const multer = require("multer");
const jwt = require('jsonwebtoken');
const path = require('path');
const fs = require('fs');
const UserAuthModel = require("../models/UserAuth");
const PatientDetailsModel = require("../models/PatientDetails");
const ModelEvaluationModel = require("../models/ModelEvaluation"); // Add this model
const auth = require('../middleware/auth');

const router = express.Router();
const cors = require("cors");

// Configure multer for file storage
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    const uploadDir = path.join(__dirname, '../uploads');
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + '-' + file.originalname);
  }
});

const upload = multer({
  storage,
  limits: { fileSize: 50 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const allowedTypes = /jpg|jpeg|png|gif/;
    const fileType = allowedTypes.test(file.mimetype);
    if (fileType) {
      cb(null, true);
    } else {
      cb(new Error("Only image files are allowed!"), false);
    }
  },
});

// Login route
router.post("/login", async (req, res) => {
  try {
    const { email, password } = req.body;

    // Find user
    const user = await UserAuthModel.findOne({ email });
    if (!user) {
      return res.status(404).json({ message: "No record found" });
    }

    // Verify password
    const isMatch = await user.comparePassword(password);
    if (!isMatch) {
      return res.status(401).json({ message: "The password is incorrect" });
    }

    // Create JWT token
    const token = jwt.sign(
      { 
        id: user._id,
        email: user.email,
        role: user.role,
        username: user.username
      },
      process.env.JWT_SECRET,
      { expiresIn: '24h' }
    );

    const response = { 
      message: "Success", 
      token,
      username: user.username, 
      role: user.role,
      _id: user._id
    };

    if (user.role === "doctor") {
      response.doctorId = user._id;
    }

    res.json(response);
  } catch (error) {
    console.error("Login error:", error);
    res.status(500).json({ message: "Server error" });
  }
});

// Signup route
router.post("/signup", async (req, res) => {
  try {
    const { username, lastName, email, password, role } = req.body;

    // Check if user already exists
    const existingUser = await UserAuthModel.findOne({ email });
    if (existingUser) {
      return res.status(400).json({ message: "Email already registered" });
    }

    // Create new user
    const user = new UserAuthModel({
      username,
      lastName,
      email,
      password,
      role
    });

    await user.save();

    // Create JWT token
    const token = jwt.sign(
      { 
        id: user._id,
        email: user.email,
        role: user.role,
        username: user.username
      },
      process.env.JWT_SECRET,
      { expiresIn: '24h' }
    );

    res.status(201).json({
      message: "User created successfully",
      token,
      username: user.username,
      role: user.role,
      _id: user._id
    });
  } catch (error) {
    console.error("Signup error:", error);
    res.status(500).json({ message: "Server error" });
  }
});

// Protected routes
router.post("/addPatient", auth, upload.array("xray", 5), async (req, res) => {
  try {
    console.log('Request body:', req.body);
    console.log('Request files:', req.files);
    console.log('User:', req.user);

    // ✅ Extract and normalize fileName
    let { name, location, age, gender, fileName } = req.body;

    // ✅ If fileName comes as an array, take the first item
    if (Array.isArray(fileName)) {
      fileName = fileName[0];
    }

    // Convert uploaded X-ray files to base64
    const xrayBase64 = [];
    if (req.files && req.files.length > 0) {
      for (const file of req.files) {
        const fileBuffer = fs.readFileSync(file.path);
        const base64 = fileBuffer.toString('base64');
        xrayBase64.push(base64);
        fs.unlinkSync(file.path); // Clean up temp file
      }
    }

    // Validate required fields
    if (!name || !location || !age || !gender || !fileName) {
      return res.status(400).json({ error: "Missing required fields" });
    }

    // Ensure user is authenticated
    if (!req.user || !req.user.id) {
      return res.status(401).json({ error: "User not authenticated" });
    }

    // ✅ Create the patient record
    const patient = new PatientDetailsModel({
      name,
      location,
      age: parseInt(age),
      gender,
      fileName: String(fileName), // Ensure it's a string
      xray: xrayBase64,
      createdBy: req.user.id
    });

    await patient.save();
    res.status(201).json(patient);
  } catch (error) {
    console.error("Error adding patient:", error);
    res.status(500).json({ error: error.message });
  }
});


router.get("/patients", auth, async (req, res) => {
  try {
    const patients = await PatientDetailsModel.find({ createdBy: req.user.id });
    res.status(200).json(patients);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

router.delete("/deletePatient/:id", auth, async (req, res) => {
  try {
    const { id } = req.params;
    
    if (!mongoose.Types.ObjectId.isValid(id)) {
      return res.status(400).json({ error: "Invalid patient ID format" });
    }

    const patient = await PatientDetailsModel.findById(id);
    
    if (!patient) {
      return res.status(404).json({ error: "Patient not found" });
    }

    // Check if user has permission to delete
    if (patient.createdBy.toString() !== req.user.id && req.user.role !== 'doctor') {
      return res.status(403).json({ error: "Not authorized to delete this patient" });
    }

    const deletedPatient = await PatientDetailsModel.findByIdAndDelete(id);
    if (!deletedPatient) {
      return res.status(404).json({ error: "Patient not found or already deleted" });
    }

    res.status(200).json({ message: "Patient deleted successfully" });
  } catch (error) {
    console.error("Error in deletePatient route:", error);
    res.status(500).json({ 
      error: "Internal server error",
      details: error.message 
    });
  }
});

// NEW ASSIGNMENT ROUTE: Update the PatientDetails record with doctorId
router.post('/assign-to-doctor', async (req, res) => {
  try {
    const { patientId, doctorId } = req.body;
    if (!mongoose.Types.ObjectId.isValid(patientId) || !mongoose.Types.ObjectId.isValid(doctorId)) {
      return res.status(400).json({ error: "Invalid patientId or doctorId format." });
    }
    const patient = await PatientDetailsModel.findById(patientId);
    if (!patient) return res.status(404).json({ error: "Patient not found." });

    // Update patient record by assigning the doctor
    patient.doctorId = doctorId;
    await patient.save();

    res.status(200).json({ message: "Patient assigned successfully." });
  } catch (err) {
    console.error("Error during assignment:", err);
    res.status(500).json({ error: err.message });
  }
});

// NEW FETCH ASSIGNED PATIENTS: Query PatientDetails by doctorId
router.get("/patients/assign-to-doctor/:doctorId", async (req, res) => {
  try {
    const { doctorId } = req.params;
    if (!mongoose.Types.ObjectId.isValid(doctorId)) {
      return res.status(400).json({ error: "Invalid doctorId" });
    }
    const assignedPatients = await PatientDetailsModel.find({ doctorId });
    res.status(200).json(assignedPatients);
  } catch (error) {
    console.error("Error fetching assigned patients:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

// NEW Update Evaluation route: Update evaluation in PatientDetails
router.put('/updateEvaluation/:id', async (req, res) => {
  try {
    const { evaluation, findings } = req.body;
    const updatedPatient = await PatientDetailsModel.findByIdAndUpdate(
      req.params.id,
      { evaluation, findings },
      { new: true }
    );
    if (!updatedPatient) {
      return res.status(404).json({ message: 'Patient not found' });
    }
    res.status(200).json({ message: 'Evaluation updated successfully', patient: updatedPatient });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// Protected dashboard routes
router.get("/dashboard-counts", auth, async (req, res) => {
  try {
    const totalDoctors = await UserAuthModel.countDocuments({ role: "doctor" });
    const totalRadtechs = await UserAuthModel.countDocuments({ role: "radtech" });
    const totalPatients = await PatientDetailsModel.countDocuments();

    res.json({ totalDoctors, totalRadtechs, totalPatients });
  } catch (error) {
    console.error("Error fetching dashboard counts:", error);
    res.status(500).json({ error: "Internal Server Error" });
  }
});

router.get("/doctors", auth, async (req, res) => {
  try {
    const doctors = await UserAuthModel.find({ role: "doctor" })
      .select("username _id")
      .lean();
    res.status(200).json(doctors);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get a single patient by ID
router.get("/patients/:id", auth, async (req, res) => {
  try {
    const patient = await PatientDetailsModel.findById(req.params.id);
    if (!patient) {
      return res.status(404).json({ message: "Patient not found" });
    }
    res.json(patient);
  } catch (error) {
    console.error("Error fetching patient:", error);
    res.status(500).json({ message: "Error fetching patient data" });
  }
});

router.post("/patients/saveResult", auth, async (req, res) => {
  try {
    const { classifiedDisease, imageSrc } = req.body;

    if (!classifiedDisease || !imageSrc) {
      return res.status(400).json({ message: "Missing classifiedDisease or imageSrc" });
    }

    // Extract base64 image data if it comes with a data URL prefix
    let imageData = imageSrc;
    if (imageSrc.includes('base64,')) {
      imageData = imageSrc.split('base64,')[1];
    }

    // Create a new Patient record (or you could update an existing one if you want)
    const newPatient = new PatientDetailsModel({
      name: `Classified Patient ${Date.now()}`,
      location: "Unknown",
      age: 0,
      gender: "Unknown",
      xray: [imageData], // Store the clean base64 data
      classifiedDisease,
      createdBy: req.user.id // the user who classified it
    });

    await newPatient.save();

    res.status(201).json({ message: "Classification result saved successfully", patient: newPatient });
  } catch (error) {
    console.error("Error saving classified result:", error);
    res.status(500).json({ message: "Server error" });
  }
});

// Update patient classification
router.put("/updateClassification/:id", auth, async (req, res) => {
  try {
    const { classifiedDisease } = req.body;
    const { id } = req.params;

    if (!classifiedDisease && classifiedDisease !== 0) {
      return res.status(400).json({ message: "Missing classifiedDisease" });
    }

    const patient = await PatientDetailsModel.findByIdAndUpdate(
      id,
      { classifiedDisease },
      { new: true }
    );

    if (!patient) {
      return res.status(404).json({ message: "Patient not found" });
    }

    res.status(200).json({ message: "Patient classification updated", patient });
  } catch (error) {
    console.error("Update classification error:", error);
    res.status(500).json({ message: "Server error" });
  }
});

router.get("/tfjs-model-info", (req, res) => {
  const modelPath = path.join(__dirname, "../static/models/tfjs/model.json");
  if (fs.existsSync(modelPath)) {
    return res.json({ available: true, url: "/models/tfjs/model.json" });
  } else {
    return res.status(404).json({ available: false });
  }
});

// Add new route for downloading files
router.get("/download/:patientId/:classification", auth, async (req, res) => {
  try {
    const { patientId, classification } = req.params;
    
    // Create classification directory if it doesn't exist
    const downloadDir = path.join(__dirname, '../downloads', classification);
    if (!fs.existsSync(downloadDir)) {
      fs.mkdirSync(downloadDir, { recursive: true });
    }

    // Get patient data
    const patient = await PatientDetailsModel.findById(patientId);
    if (!patient || !patient.xray || patient.xray.length === 0) {
      return res.status(404).json({ message: "No X-ray images found" });
    }

    // Save images to the classification directory
    const savedFiles = [];
    for (let i = 0; i < patient.xray.length; i++) {
      const imageBuffer = Buffer.from(patient.xray[i], 'base64');
      const fileName = `xray_image_${i + 1}.jpg`;
      const filePath = path.join(downloadDir, fileName);
      
      fs.writeFileSync(filePath, imageBuffer);
      savedFiles.push(filePath);
    }

    res.json({ 
      message: "Files saved successfully",
      files: savedFiles,
      downloadPath: downloadDir
    });
  } catch (error) {
    console.error("Download error:", error);
    res.status(500).json({ message: "Error saving files" });
  }
});



// NEW ROUTE: API endpoint to process an image for model prediction
router.post("/models/process-image", auth, upload.single("image"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ message: "No image file provided" });
    }

    // Read the uploaded file
    const imageBuffer = fs.readFileSync(req.file.path);
    const base64Image = imageBuffer.toString('base64');
    
    // Clean up the temporary file
    fs.unlinkSync(req.file.path);
    
    // Return the base64 encoded image for client-side processing
    res.status(200).json({ 
      message: "Image processed successfully",
      imageData: base64Image
    });
  } catch (error) {
    console.error("Error processing image:", error);
    res.status(500).json({ message: "Error processing image" });
  } 
});

router.put('/modelEval/:id', async (req, res) => {
  try {
    const { evaluation, findings, modelevaluation } = req.body;
    const updates = {};
    if (evaluation      != null) updates.evaluation      = evaluation;
    if (findings        != null) updates.findings        = findings;
    if (modelevaluation != null) updates.modelevaluation = modelevaluation;
    console.log('Applying updates for', req.params.id, updates);
    const updated = await PatientDetailsModel.findByIdAndUpdate(
      req.params.id,
      updates,
      { new: true }
    );
    if (!updated) return res.status(404).json({ message: 'Patient not found' });
    res.json({ message: 'Updated!', patient: updated });
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: err.message });
  }
});

router.get('/modelEvaluation/:id', async (req, res) => {
  try {
    const patientId = req.params.id;
    
    // Find the patient by ID
    const patient = await PatientDetailsModel.findById(patientId);
    
    if (!patient) {
      return res.status(404).json({ 
        success: false, 
        message: 'Patient not found' 
      });
    }

    // Check if there's model evaluation data
    if (patient.modelevaluation === undefined) {
      return res.status(200).json({
        success: true,
        message: 'No evaluation data available for this patient',
        evaluation: null
      });
    }

    // Return the evaluation data with actual values from the patient
    const evaluationData = {
      // Include any evaluation metrics if you have stored them separately
      last_updated: new Date(),
      
      // Include the actual classification result and evaluation
      modelevaluation: patient.modelevaluation, // Store the numeric class ID
      evaluation: patient.evaluation // Store the text diagnosis
    };

    return res.status(200).json({
      success: true,
      evaluation: evaluationData
    });
  } catch (err) {
    console.error('Error fetching evaluation data:', err);
    return res.status(500).json({
      success: false,
      message: 'Server error while fetching evaluation data',
      error: err.message
    });
  }
});


module.exports = router;