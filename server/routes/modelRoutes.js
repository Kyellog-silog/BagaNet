const express = require("express");
const mongoose = require("mongoose");
const ModelEvaluationModel = require("../models/ModelEvaluation");
const auth = require('../middleware/auth');

const router = express.Router();


router.get("/evaluation/baganet", auth, async (req, res) => {
    try {
        const evaluation = await ModelEvaluationModel.findOne({ modelName: 'BAGANET' });
        
        if (!evaluation) {
        return res.status(404).json({ message: "No evaluation data found for BAGANET model" });
        }
        
        res.status(200).json({ evaluation });
    } catch (error) {
        console.error("Error fetching model evaluation:", error);
        res.status(500).json({ message: "Server error" });
    }
});


router.post("/evaluation", auth, async (req, res) => {
    try {
        const { modelName, metrics, version } = req.body;
        
        if (!modelName || !metrics) {
        return res.status(400).json({ message: "Missing required fields" });
        }
        
        // Check if user has admin permission to update model data
        if (req.user.role !== 'admin') {
        return res.status(403).json({ message: "Not authorized to update model evaluation data" });
        }
        
        let evaluation = await ModelEvaluationModel.findOne({ modelName });
        
        if (evaluation) {
        // Update existing evaluation
        evaluation.metrics = metrics;
        evaluation.version = version || evaluation.version;
        evaluation.updatedAt = new Date();
        await evaluation.save();
        } else {
        // Create new evaluation
        evaluation = new ModelEvaluationModel({
            modelName,
            metrics,
            version: version || '1.0',
            createdBy: req.user.id
        });
        await evaluation.save();
        }
        
        res.status(200).json({ message: "Model evaluation data saved", evaluation });
    } catch (error) {
        console.error("Error saving model evaluation:", error);
        res.status(500).json({ message: "Server error" });
    }
});

module.exports = router;