const UserAuthModel = require('../models/UserAuth');

const getDoctors = async (req, res) => {
  try {
    const doctors = await UserAuthModel.find({ role: 'doctor' })
      .select('username lastName email role')
      .lean();
    
    res.json(doctors);
  } catch (error) {
    console.error('Error fetching doctors:', error);
    res.status(500).json({ message: 'Error fetching doctors' });
  }
};

module.exports = {
  getDoctors
}; 