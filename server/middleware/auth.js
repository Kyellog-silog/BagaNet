const jwt = require('jsonwebtoken');

const auth = (req, res, next) => {
  try {
    const token = req.header('Authorization')?.replace('Bearer ', '');
    
    if (!token) {
      return res.status(401).json({ message: 'No authentication token, access denied' });
    }

    const verified = jwt.verify(token, process.env.JWT_SECRET);
    console.log('Verified token payload:', verified);

    // Ensure all required user fields are present
    if (!verified.id) {
      return res.status(401).json({ message: 'Invalid token payload: missing user ID' });
    }

    // Map the token payload to req.user
    req.user = {
      id: verified.id.toString(), // Ensure id is a string
      role: verified.role || 'radtech', // Default to radtech if role is missing
      email: verified.email,
      username: verified.username
    };

    console.log('Mapped user:', req.user);
    next();
  } catch (err) {
    console.error('Auth middleware error:', err);
    res.status(401).json({ message: 'Token verification failed, authorization denied' });
  }
};

module.exports = auth; 