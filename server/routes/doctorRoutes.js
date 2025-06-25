const express = require('express');
const router = express.Router();
const { getDoctors } = require('../controllers/doctorController');
const auth = require('../middleware/auth');

// Get all doctors
router.get('/', auth, getDoctors);

module.exports = router; 