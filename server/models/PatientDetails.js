const mongoose = require("mongoose");

const PatientDetailsSchema = new mongoose.Schema({
  name: { type: String, required: true },
  location: { type: String, required: true },
  age: { type: Number, required: true },
  gender: { type: String, required: true },
  xray: { type: [String], required: false },
  doctorId: { type: mongoose.Schema.Types.ObjectId, ref: 'Doctor' },
  evaluation: { type: String, required: false },
  modelevaluation: { type: String, required: false },
  modelDiagnosisText: { type: String, required: false },
  findings: { type: String, required: false },
  classifiedDisease: { type: String, required: false},
  fileName: { type: String, required: true },
  createdBy: { type: mongoose.Schema.Types.ObjectId, ref: 'UserAuth', required: true },
  trainingStatus: {
    type: String,
    enum: ['unused', 'used', 'purged'],
    default: 'unused'
  },
  lastTrainingDate: {
    type: Date,
    default: null
  }
});

const PatientDetailsModel = mongoose.model("PatientDetails", PatientDetailsSchema);
module.exports = PatientDetailsModel;