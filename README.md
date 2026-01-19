# Performance Assessment Application

A full-stack web application for conducting and managing employee performance assessments using interactive polar area charts. Features secure authentication, MongoDB persistence, and comprehensive assessment history management.

## 📋 Features

- **Interactive Polar Area Charts**: Visual representation of 19 performance metrics across 4 themes
- **Secure Authentication**: JWT-based user registration and login
- **User Management**: Admin users can create, edit, and delete user accounts
- **Role-Based Access**: Two roles - `assessor` (default) and `admin`
- **MongoDB Persistence**: Auto-save assessments every 5 seconds with offline fallback
- **Assessment History**: View, search, filter, edit, and delete past assessments
- **Multi-User Support**: Each user sees only their own assessments
- **CSV Import/Export**: Import and export assessment data
- **Responsive Design**: Works on desktop and mobile devices
- **Real-time Updates**: Chart updates instantly as you change values

### Assessment Themes & Metrics

1. **Strategic Vision and Business Alignment** (4 metrics)
   - Shared Vision, Strategy, Business Alignment, Customer Focus

2. **Focus and Engagement** (5 metrics)
   - Cross-functional Teams, Clarity of Priorities, Acceptance Criteria, Focus, Engagement

3. **Autonomy and Change** (5 metrics)
   - Feedback, Enabling Autonomy, Change and Ambiguity, Desired Culture, Works Autonomously

4. **Stakeholders and Team** (5 metrics)
   - Stakeholders, Team Attrition, Team Focus, Developing People, Prepare Subordinates for Success

## 🚀 Production Deployment

**Deploying to Ubuntu Server?** See the comprehensive **[DEPLOYMENT.md](DEPLOYMENT.md)** guide for:
- Complete Ubuntu server setup
- MongoDB configuration and security
- **.env file security** (permissions, encryption)
- PM2 process management
- Nginx reverse proxy configuration
- SSL/HTTPS setup with Let's Encrypt
- Firewall configuration (UFW)
- Security hardening
- Automated backups
- Monitoring and logging

**Quick Deploy:**
```bash
sudo ./deploy.sh
```

Then follow the post-installation steps in [DEPLOYMENT.md](DEPLOYMENT.md).

## 🛠️ Prerequisites

Before you begin, ensure you have the following installed:

- **Node.js** (v14 or higher) - [Download](https://nodejs.org/)
- **MongoDB** (v4.4 or higher) - [Download](https://www.mongodb.com/try/download/community)
- **npm** (comes with Node.js)
- A modern web browser (Chrome, Firefox, Safari, Edge)

## 📦 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd performancePolarArea
```

### 2. Install Dependencies

```bash
npm install
```

This will install all backend dependencies including:
- Express (web framework)
- Mongoose (MongoDB ODM)
- JWT (authentication)
- bcryptjs (password hashing)
- And more...

### 3. Configure Environment Variables

Create a `.env` file in the root directory (copy from `.env.example`):

```bash
cp .env.example .env
```

Edit `.env` and configure your settings:

```env
# Server Configuration
NODE_ENV=development
PORT=5000
FRONTEND_URL=http://localhost:3000

# MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017/hr_performance
MONGODB_DB_NAME=hr_performance

# JWT Configuration
JWT_SECRET=your-secret-key-here
JWT_EXPIRES_IN=24h

# Security
BCRYPT_ROUNDS=12

# Registration Control
ALLOW_REGISTRATION=true
```

**Important:** Generate a strong JWT secret:
```bash
node -e "console.log(require('crypto').randomBytes(64).toString('hex'))"
```

### 4. Set Up MongoDB

#### Option A: Local MongoDB

1. Start MongoDB service:
   ```bash
   # macOS (Homebrew)
   brew services start mongodb-community

   # Linux (systemd)
   sudo systemctl start mongod

   # Windows
   # Start MongoDB as a Windows service or run mongod.exe
   ```

2. Verify MongoDB is running:
   ```bash
   mongosh
   # Should connect successfully
   ```

#### Option B: MongoDB Atlas (Cloud)

1. Create a free account at [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
2. Create a cluster
3. Get your connection string
4. Update `MONGODB_URI` in `.env`:
   ```env
   MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/hr_performance
   ```

## 🚀 Running the Application

### Start the Backend Server

```bash
npm start
```

You should see:
```
✓ Server running on port 5000
✓ Environment: development
✓ Frontend URL: http://localhost:3000
✓ MongoDB connected: hr_performance
```

### Start the Frontend Server

In a **separate terminal**, run:

```bash
python3 -m http.server 3000
```

Or use any static file server:

```bash
# Using npx
npx http-server -p 3000

# Using Python 2
python -m SimpleHTTPServer 3000
```

### Access the Application

Open your browser and navigate to:

**http://localhost:3000/login.html**

## 📖 Usage Guide

### 1. Register a New Account

- Navigate to http://localhost:3000/login.html
- Click "Create one" to show registration form
- Enter your details:
  - First Name
  - Last Name
  - Email
  - Password (minimum 6 characters)
  - Confirm Password
- Click "Create Account"

### 2. Login

- Enter your email and password
- Click "Sign In"
- You'll be redirected to the assessment page

### 3. Create an Assessment

- Enter the employee's name
- Rate each metric from 0-5:
  - 0 = Not Applicable
  - 1 = Very Poor
  - 2 = Poor
  - 3 = Fair
  - 4 = Good
  - 5 = Excellent
- Watch the polar area chart update in real-time
- Assessment auto-saves to MongoDB every 5 seconds

### 4. View Assessment History

- Click the "View History" button at the bottom
- See all your saved assessments in a table
- Filter by employee name or date range
- View average scores for each assessment

### 5. Edit an Existing Assessment

- From the history page, click "Edit" on any assessment
- Make your changes
- Press Ctrl+S (or Cmd+S on Mac) to save immediately
- Or wait 5 seconds for auto-save

### 6. Delete an Assessment

- From the history page, click "Delete"
- Confirm the deletion
- Assessment is permanently removed

### 7. Export to CSV

- Click "Save to CSV" button
- Downloads a CSV file with all ratings
- Can be opened in Excel or imported later

### 8. Import from CSV

- Click "Load from CSV" button
- Select a previously exported CSV file
- Data is loaded and saved to MongoDB

### 9. User Settings

- Click "Settings" in the navigation bar
- Update your profile (name, email)
- Change your password (requires current password)

### 10. User Management (Admin Only)

Admin users have access to the User Management section:

- **View all users**: See a table of all registered users
- **Add new user**: Create accounts with email, password, name, and role
- **Edit user**: Update any user's information, including role
- **Delete user**: Remove users (also deletes their assessments)

**Security protections:**
- Cannot delete your own account
- Cannot delete the last admin user
- Cannot demote yourself from admin if you're the last one

## 🔐 Controlling User Registration

By default, anyone can create a new account. For production environments or when you want to restrict who can access the system, you can disable new user registration.

### Disable Registration

Edit your `.env` file and set:

```env
ALLOW_REGISTRATION=false
```

Then restart the backend server:

```bash
npm start
```

**What happens when registration is disabled:**
- The "Create one" link is hidden on the login page
- The registration form is not accessible
- API registration endpoint returns 403 Forbidden
- Login page shows: "Registration is currently disabled. Please contact an administrator."
- **Existing users can still login** normally

### Creating Users When Registration is Disabled

If you need to create user accounts when registration is disabled, you have two options:

**Option 1: Temporarily Enable Registration**
1. Set `ALLOW_REGISTRATION=true` in `.env`
2. Restart backend server
3. Create the user account(s)
4. Set `ALLOW_REGISTRATION=false` and restart

**Option 2: Create via MongoDB Shell**

```javascript
// Connect to MongoDB
mongosh hr_performance

// Insert user (password will need to be bcrypt hashed manually)
db.users.insertOne({
  email: "admin@example.com",
  password: "$2a$12$hashedPasswordHere",  // Use bcrypt to hash
  firstName: "Admin",
  lastName: "User",
  role: "assessor",
  createdAt: new Date()
});
```

**To generate a bcrypt hash for a password:**

```bash
node -e "const bcrypt = require('bcryptjs'); bcrypt.hash('yourpassword', 12).then(hash => console.log(hash));"
```

## 👥 User Roles

The application supports two user roles:

| Role | Permissions |
|------|-------------|
| `assessor` | Create/view/edit/delete own assessments, update own profile |
| `admin` | All assessor permissions + manage all users (create, edit, delete) |

### Making a User an Admin

New users are created with the `assessor` role by default. To promote a user to admin:

**Option 1: Via Admin UI**
- Login as an existing admin
- Go to Settings → User Management
- Click "Edit" on the user
- Change role to "Admin"

**Option 2: Via MongoDB**
```javascript
mongosh hr_performance
db.users.updateOne(
  { email: "user@example.com" },
  { $set: { role: "admin" } }
)
```

**Option 3: First User Setup**
If this is a fresh installation and you need to create the first admin:
```javascript
mongosh hr_performance
db.users.updateOne(
  { email: "your-email@example.com" },
  { $set: { role: "admin" } }
)
```

## 📁 Project Structure

```
performancePolarArea/
├── backend/                    # Node.js/Express backend
│   ├── config/
│   │   ├── database.js        # MongoDB connection
│   │   └── jwt.js             # JWT utilities
│   ├── controllers/
│   │   ├── authController.js  # Auth logic
│   │   └── assessmentController.js
│   ├── middleware/
│   │   ├── authMiddleware.js  # JWT verification
│   │   └── validation.js      # Input validation
│   ├── models/
│   │   ├── User.js            # User schema
│   │   └── Assessment.js      # Assessment schema
│   ├── routes/
│   │   ├── auth.js            # Auth endpoints
│   │   └── assessments.js     # Assessment endpoints
│   └── server.js              # Express app entry
│
├── frontend/                   # Frontend code
│   └── api/
│       ├── apiClient.js       # API wrapper
│       └── authManager.js     # Session management
│
├── index-std.html             # Standard assessment page
├── index-3t.html              # 3T branded version
├── login.html                 # Login/register page
├── history.html               # Assessment history page
├── user-config.html           # User settings & admin page
├── user-config.js             # User settings functionality
├── app.js                     # Assessment app (module)
├── script.js                  # Assessment app (standalone)
├── config.js                  # App configuration
├── styles.css                 # Global styles
├── package.json               # Dependencies
├── .env                       # Environment config
└── README.md                  # This file
```

## 🔌 API Endpoints

### Authentication

- `GET /api/auth/config` - Get authentication configuration (public)
  - Returns: `{ allowRegistration: boolean }`
- `POST /api/auth/register` - Create new user account
- `POST /api/auth/login` - Login and get JWT token
- `POST /api/auth/logout` - Logout user
- `GET /api/auth/me` - Get current user info (protected)

### User Management (Admin Only)

- `GET /api/auth/users` - List all users (admin only)
- `POST /api/auth/users` - Create new user (admin only)
  - Body: `{ email, password, firstName, lastName, role? }`
- `PUT /api/auth/users/:id` - Update user (self or admin)
  - Body: `{ email?, password?, firstName?, lastName?, role?, currentPassword? }`
  - Note: Regular users need `currentPassword` to change email or password
- `DELETE /api/auth/users/:id` - Delete user (admin only)
  - Cascade: Also deletes user's assessments

### Assessments

All assessment endpoints require authentication (JWT token).

- `GET /api/assessments` - List user's assessments
  - Query params: `page`, `limit`, `employeeName`, `dateFrom`, `dateTo`
- `GET /api/assessments/:id` - Get single assessment
- `POST /api/assessments` - Create new assessment
- `PUT /api/assessments/:id` - Update assessment
- `DELETE /api/assessments/:id` - Delete assessment
- `POST /api/assessments/import-csv` - Import from CSV
- `GET /api/assessments/export-csv` - Export to CSV

## 🔧 Technologies Used

### Frontend
- HTML5, CSS3, JavaScript (ES6+)
- [Chart.js](https://www.chartjs.org/) - Polar area charts
- Fetch API - HTTP requests
- LocalStorage - Offline fallback

### Backend
- [Node.js](https://nodejs.org/) - Runtime
- [Express](https://expressjs.com/) - Web framework
- [MongoDB](https://www.mongodb.com/) - Database
- [Mongoose](https://mongoosejs.com/) - ODM
- [JWT](https://jwt.io/) - Authentication
- [bcryptjs](https://github.com/dcodeIO/bcrypt.js) - Password hashing
- [express-validator](https://express-validator.github.io/) - Input validation
- [Helmet](https://helmetjs.github.io/) - Security headers
- [CORS](https://github.com/expressjs/cors) - Cross-origin support

## 🔒 Security Features

- **Password Hashing**: bcrypt with cost factor 12
- **JWT Tokens**: 24-hour expiry, secure secret
- **Rate Limiting**: 5 login attempts per 15 minutes
- **Input Validation**: All inputs validated server-side
- **XSS Prevention**: Input sanitization
- **CORS**: Restricted to frontend origin only
- **User Isolation**: Users can only access their own data

## 🧪 Testing

Test the API endpoints using curl:

```bash
# Register a user
curl -X POST http://localhost:5000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"test123","firstName":"Test","lastName":"User"}'

# Login
curl -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"test123"}'

# Get assessments (replace TOKEN with actual JWT)
curl http://localhost:5000/api/assessments \
  -H "Authorization: Bearer TOKEN"
```

## 🐛 Troubleshooting

### Backend won't start

**Error:** "Cannot connect to MongoDB"
- **Solution:** Ensure MongoDB is running: `mongosh` should connect successfully
- Check `MONGODB_URI` in `.env` is correct

**Error:** "Port 5000 already in use"
- **Solution:** Change `PORT` in `.env` or kill the process using port 5000

### Frontend can't connect to backend

**Error:** CORS errors in browser console
- **Solution:** Ensure `FRONTEND_URL` in `.env` matches your frontend server port
- Restart backend after changing `.env`

### Can't login / Registration fails

**Error:** "Network request failed"
- **Solution:** Check that backend is running on port 5000
- Open http://localhost:5000/health - should return `{"status":"ok"}`

**Error:** "Validation error"
- **Solution:** Password must be at least 6 characters
- Email must be valid format

### Auto-save not working

- Check browser console for errors
- Ensure you're logged in (check top-right corner for your name)
- Check backend logs for API errors
- If offline, data saves to localStorage as fallback

## 📝 Development Notes

### File Versions

- `index-std.html` + `app.js` - Uses ES6 modules, requires web server
- `index-3t.html` + inline script - Standalone, works with `file://` protocol
- Both versions have identical functionality

### Auto-Save Behavior

- Debounced 5-second delay after last change
- Saves to MongoDB with localStorage backup
- Creates new assessment on first save
- Updates existing assessment on subsequent saves

### Offline Mode

- If backend is unavailable, saves to localStorage
- Shows message: "Saved locally (offline mode)"
- Data syncs to MongoDB when connection restored

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## 📄 License

This project is for internal use. All rights reserved.

## 👤 Author

Peter Caron Consulting OÜ

## 🙏 Acknowledgments

- Chart.js for the excellent charting library
- 3T SoftwareLabs GmbH for the branded version
- MongoDB for the database platform

---

**Need help?** Check the troubleshooting section above or review the server logs for detailed error messages.
