# VisiML Deployment Guide

## 🚀 Quick Deploy Options

### Option 1: Streamlit Community Cloud (Recommended - FREE)
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub and select this repo
4. Click "Deploy!"
5. Your app will be live at: `https://your-app-name.streamlit.app`

### Option 2: Render (FREE tier available)
1. Create account at [render.com](https://render.com)
2. Connect GitHub repo
3. Choose "Web Service"
4. Use these settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
   - **Environment**: Python 3

### Option 3: Heroku
1. Install Heroku CLI
2. Run these commands:
```bash
heroku create your-app-name
git add .
git commit -m "Deploy VisiML"
git push heroku main
```

## 📁 Files Added for Deployment
- `render.yaml` - Render configuration
- `Procfile` - Process file for Heroku
- `runtime.txt` - Python version specification
- `.streamlit/config.toml` - Streamlit configuration
- Updated `requirements.txt` - Deployment-ready dependencies

## 🔧 Environment Variables (if needed)
No special environment variables required for basic deployment.

## 📝 Notes
- The app uses matplotlib, which works well on all these platforms
- Free tiers have some limitations (sleep after inactivity)
- For production use, consider paid tiers for better performance

## 🎯 Recommended: Streamlit Community Cloud
It's specifically designed for Streamlit apps and offers:
- ✅ Free hosting
- ✅ Automatic deployments from GitHub
- ✅ Easy setup
- ✅ Good performance for ML apps
- ✅ Built-in SSL
