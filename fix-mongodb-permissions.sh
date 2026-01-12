#!/bin/bash
################################################################################
# Fix MongoDB Permissions for Assessment Writing
#
# This script checks and fixes MongoDB permissions for the application user
################################################################################

echo "MongoDB Permissions Diagnostic and Fix"
echo "======================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Step 1: Checking if you can connect to MongoDB...${NC}"
mongosh --version > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: mongosh not found. Install MongoDB shell first.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ mongosh installed${NC}"
echo ""

echo -e "${YELLOW}Step 2: Testing MongoDB connection...${NC}"
echo ""
echo "Please enter your MongoDB application user password when prompted:"
echo ""

# Test connection
mongosh "mongodb://perfassess:YOUR_PASSWORD@localhost:27017/hr_performance?authSource=hr_performance" --eval "db.runCommand({ping:1})" > /dev/null 2>&1

if [ $? -ne 0 ]; then
    echo -e "${RED}Cannot connect with current credentials.${NC}"
    echo ""
    echo "Let's fix the MongoDB user permissions."
    echo ""

    echo -e "${YELLOW}Connecting to MongoDB as admin to fix permissions...${NC}"

    # Create a MongoDB script to run
    cat > /tmp/fix_mongodb.js << 'MONGOEOF'
// Switch to hr_performance database
use hr_performance

// Check if user exists
const user = db.getUser("perfassess");
if (user) {
    print("User perfassess exists");
    print("Current roles: " + JSON.stringify(user.roles));

    // Drop and recreate with correct permissions
    print("\nDropping user...");
    db.dropUser("perfassess");
}

// Create user with full read/write permissions
print("\nCreating user with readWrite permissions...");
db.createUser({
  user: "perfassess",
  pwd: "12gramscI",
  roles: [
    { role: "readWrite", db: "hr_performance" }
  ]
});

print("\n✓ User created successfully");

// Test permissions
print("\nTesting write permission...");
db.test_collection.insertOne({test: "write_test", timestamp: new Date()});
print("✓ Write test successful");

print("\nTesting read permission...");
const count = db.test_collection.countDocuments({test: "write_test"});
print("✓ Read test successful (found " + count + " documents)");

// Clean up test
db.test_collection.deleteMany({test: "write_test"});
print("✓ Cleanup successful");

// Show final user permissions
print("\nFinal user configuration:");
const finalUser = db.getUser("perfassess");
print("User: " + finalUser.user);
print("Roles: " + JSON.stringify(finalUser.roles, null, 2));
MONGOEOF

    echo ""
    echo "Running MongoDB fix script..."
    echo "You'll be prompted for the MongoDB admin password."
    echo ""

    sudo mongosh -u admin -p --authenticationDatabase admin < /tmp/fix_mongodb.js

    rm /tmp/fix_mongodb.js

    echo ""
    echo -e "${GREEN}MongoDB user permissions updated!${NC}"
    echo ""

else
    echo -e "${GREEN}✓ Connection successful${NC}"
    echo ""
fi

echo -e "${YELLOW}Step 3: Checking current user permissions...${NC}"
echo ""

# Check permissions
mongosh "mongodb://perfassess:YOUR_PASSWORD@localhost:27017/hr_performance?authSource=hr_performance" --eval "
const user = db.getUser('perfassess');
print('User: ' + user.user);
print('Roles:');
user.roles.forEach(role => {
    print('  - Role: ' + role.role + ' on DB: ' + role.db);
});
" 2>/dev/null

echo ""
echo -e "${YELLOW}Step 4: Testing write permissions...${NC}"

mongosh "mongodb://perfassess:YOUR_PASSWORD@localhost:27017/hr_performance?authSource=hr_performance" --eval "
try {
    // Try to write to assessments collection
    const result = db.assessments.insertOne({
        test: true,
        timestamp: new Date(),
        note: 'Permission test - will be deleted'
    });
    print('✓ Write test successful');
    print('  Inserted ID: ' + result.insertedId);

    // Clean up
    db.assessments.deleteOne({_id: result.insertedId});
    print('✓ Cleanup successful');

} catch (error) {
    print('✗ Write test failed: ' + error.message);
}
" 2>/dev/null

echo ""
echo -e "${YELLOW}Step 5: Recommendations${NC}"
echo ""
echo "After fixing permissions, you need to:"
echo "1. Update .env file with the correct MongoDB password"
echo "2. Restart the application: pm2 restart performance-assessment"
echo "3. Test creating an assessment in the browser"
echo ""
echo "Commands:"
echo "  sudo vim /var/www/performance-assessment/.env"
echo "  # Update MONGODB_URI with the password you set above"
echo "  pm2 restart performance-assessment"
echo ""

exit 0
