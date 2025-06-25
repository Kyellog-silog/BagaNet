const mongoose = require('mongoose');

const ModelEvaluationSchema = new mongoose.Schema({
    modelName: {
        type: String,
        required: true,
        unique: true
    },
    version: {
        type: String,
        default: '1.0'
    },
    metrics: {
        accuracy: {
        type: Number,
        default: 0
        },
        precision: {
        type: Number,
        default: 0
        },
        recall: {
        type: Number,
        default: 0
        },
        f1Score: {
        type: Number,
        default: 0
        },
        confusionMatrix: {
        type: mongoose.Schema.Types.Mixed,
        default: {}
        },
        classAccuracies: {
        type: Map,
        of: Number,
        default: {}
        },
        additionalMetrics: {
        type: mongoose.Schema.Types.Mixed,
        default: {}
        }
    },
    createdBy: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'UserAuth'
    },
    createdAt: {
        type: Date,
        default: Date.now
    },
    updatedAt: {
        type: Date,
        default: Date.now
    }
});

// Pre-save middleware to update the updatedAt field
ModelEvaluationSchema.pre('save', function(next) {
    this.updatedAt = Date.now();
    next();
});

module.exports = mongoose.model('ModelEvaluation', ModelEvaluationSchema);