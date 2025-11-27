set -e

echo "=============================================="
echo "  Earthquake Magnitude Predictor Deployment"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo -e "${YELLOW}Initializing git repository...${NC}"
    git init
fi

# Add all files
echo -e "${YELLOW}Adding files to git...${NC}"
git add .

# Commit changes
echo -e "${YELLOW}Committing changes...${NC}"
read -p "Enter commit message: " commit_msg
git commit -m "$commit_msg" || echo "Nothing to commit"

# Check if remote exists
if ! git remote | grep -q "origin"; then
    echo -e "${YELLOW}No remote found.${NC}"
    read -p "Enter GitHub repository URL: " repo_url
    git remote add origin "$repo_url"
fi

# Push to GitHub
echo -e "${YELLOW}Pushing to GitHub...${NC}"
git push -u origin main || git push -u origin master

echo ""
echo -e "${GREEN}=============================================="
echo "  Code pushed to GitHub!"
echo "==============================================${NC}"
echo ""
echo "Next steps for Streamlit Cloud deployment:"
echo "1. Go to https://share.streamlit.io"
echo "2. Click 'New app'"
echo "3. Select your repository"
echo "4. Set main file path: app/streamlit_app.py"
echo "5. Click 'Deploy!'"
echo ""
echo -e "${YELLOW}Note: Make sure your model files are committed!${NC}"
echo "Required files:"
echo "  - models/gp_model.pkl"
echo "  - models/scaler.pkl"
echo "  - data/earthquake.db (or run setup.py on cloud)"