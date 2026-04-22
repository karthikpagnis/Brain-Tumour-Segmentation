# 🚀 DEPLOYMENT GUIDE

Complete guide for deploying Brain Tumor Segmentation API to Heroku with online submission setup.

---

## 📋 TABLE OF CONTENTS

1. [Prerequisites](#prerequisites)
2. [Local Testing](#local-testing)
3. [GitHub Setup](#github-setup)
4. [Heroku Deployment](#heroku-deployment)
5. [Video Demo](#video-demo)
6. [Submit to Teacher](#submit-to-teacher)

---

## ✅ PREREQUISITES

You need:
- ✅ `best_model.pth` (260 MB) - from Kaggle training
- ✅ GitHub account (free)
- ✅ Heroku account (free)
- ✅ Git installed on your machine
- ✅ `app.py` file (provided)

---

## 🏠 LOCAL TESTING (Before Heroku)

Test the app locally first to ensure it works.

### Step 1: Install Dependencies

```bash
cd Brain-Tumour-Segmentation
pip install -r requirements.txt
```

### Step 2: Add Model File

Place your `best_model.pth` in the root directory:

```
Brain-Tumour-Segmentation/
├── app.py
├── best_model.pth          ← Add this file here
├── requirements.txt
├── Dockerfile
└── ...
```

### Step 3: Run Locally

```bash
python app.py
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
✅ Model ready for inference
```

### Step 4: Test in Browser

Open: `http://localhost:8000`

You should see:
- Web interface with upload area
- Model info (Dice 0.6189, etc.)
- Upload file button

### Step 5: Test Prediction

1. Prepare a test `.nii.gz` MRI file (4 channels)
2. Upload via the web interface
3. Download the result
4. Verify it worked ✅

---

## 📤 GITHUB SETUP

Push your code to GitHub for the teacher to review.

### Step 1: Create GitHub Repository

```bash
# Initialize git
cd Brain-Tumour-Segmentation
git init

# Add files
git add -A

# Create initial commit
git commit -m "Initial commit: Brain tumor segmentation API"

# Add remote (replace YOUR-USERNAME)
git remote add origin https://github.com/YOUR-USERNAME/Brain-Tumour-Segmentation.git
git branch -M main
git push -u origin main
```

### Step 2: Verify on GitHub

Visit: `https://github.com/YOUR-USERNAME/Brain-Tumour-Segmentation`

You should see:
- ✅ app.py
- ✅ requirements.txt
- ✅ Dockerfile
- ✅ README.md
- ✅ models/ folder
- ✅ data/ folder

### Step 3: Add README

Make sure `README.md` includes:

```markdown
# Brain Tumor Segmentation

**Live Demo:** https://your-app-name.herokuapp.com/

## Quick Start

1. Visit the link above
2. Upload a `.nii.gz` MRI file
3. Download segmentation prediction

## Model Info

- **Architecture:** Attention-Enhanced U-Net 3D
- **Best Dice:** 0.6189 (30 epochs)
- **Dataset:** BraTS 2019 (335 cases)
- **Classes:** Background, Necrotic Core, Edema, Enhancing Tumor

## Local Testing

```bash
pip install -r requirements.txt
python app.py
# Visit http://localhost:8000
```
```

---

## 🚀 HEROKU DEPLOYMENT (15 minutes)

Deploy your app to Heroku for a persistent URL your teacher can access.

### Step 1: Create Heroku Account

1. Go to https://www.heroku.com/
2. Sign up (free)
3. Verify email

### Step 2: Install Heroku CLI

**macOS:**
```bash
brew tap heroku/brew && brew install heroku
```

**Windows:**
Download from: https://devcenter.heroku.com/articles/heroku-cli

**Linux:**
```bash
curl https://cli-assets.heroku.com/install.sh | sh
```

### Step 3: Login to Heroku

```bash
heroku login
```

This opens a browser → click "Log in" → return to terminal

### Step 4: Create Heroku App

```bash
heroku create your-app-name
```

Replace `your-app-name` with something unique (e.g., `brain-tumor-seg-2024`)

**Output should be:**
```
Creating ⬢ brain-tumor-seg-2024... done
https://brain-tumor-seg-2024.herokuapp.com/ | git@heroku.com:brain-tumor-seg-2024.git
```

**Save this URL** - this is what you send to your teacher!

### Step 5: Add Git Remote

```bash
heroku git:remote -a your-app-name
```

### Step 6: Push to Heroku

```bash
git push heroku main
```

**Wait 3-5 minutes** for deployment to complete.

You'll see:
```
Building source
Running build process
Slug compiled to Slug file
Launching... done
```

### Step 7: View Logs

```bash
heroku logs --tail
```

Should show:
```
✅ Model loaded from best_model.pth
✅ Model ready for inference
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 8: Test Live App

Visit: `https://your-app-name.herokuapp.com/`

You should see the web interface live! ✅

---

## 🎬 VIDEO DEMO (Optional but Recommended)

Record a 5-minute video showing:

### What to Show:

1. **Open the app**
   - Visit https://your-app-name.herokuapp.com/
   - Show model info (Dice 0.6189)

2. **Upload a test MRI**
   - Prepare a `.nii.gz` file
   - Upload via drag-and-drop
   - Wait for prediction

3. **Download result**
   - Show the download link
   - Explain the output file

4. **Show GitHub repo**
   - Visit GitHub link
   - Show code structure

### Recording Software (Free):

- **macOS:** QuickTime Player (built-in)
- **Windows:** OBS Studio (free)
- **Linux:** OBS Studio (free)

### Upload Video:

- **YouTube** (unlisted)
- **Google Drive** (shareable link)
- **GitHub** (in releases)

---

## 📨 SUBMIT TO TEACHER

When submitting, send:

```
Subject: Brain Tumor Segmentation - Final Submission

Dear Professor,

Please find my submission:

1. **GitHub Repository:**
   https://github.com/YOUR-USERNAME/Brain-Tumour-Segmentation

2. **Live Demo (Test the App):**
   https://your-app-name.herokuapp.com/

3. **Demo Video:**
   https://youtu.be/your-video-id

4. **Model Performance:**
   - Architecture: Attention-Enhanced U-Net 3D
   - Best Dice: 0.6189 (validation)
   - Training: 30 epochs on BraTS 2019 dataset
   - Classes: 4 tumor regions

**You can:**
✅ Upload any MRI file and get segmentation
✅ Review all code on GitHub
✅ Run locally: python app.py

Thank you!
```

---

## 🆘 TROUBLESHOOTING

### "Model not loading"

```bash
# Check if best_model.pth exists in root
ls -lh best_model.pth

# Upload to Heroku if missing
git add best_model.pth
git commit -m "Add model file"
git push heroku main
```

### "App crashes"

```bash
# Check logs
heroku logs --tail

# Rebuild
heroku rebuild
```

### "Port already in use (local)"

```bash
# Kill process on port 8000
lsof -i :8000
kill -9 <PID>

# Or use different port
PORT=8001 python app.py
```

### "File too large for GitHub"

If `best_model.pth` is too large:

1. Use Git LFS:
   ```bash
   git lfs install
   git lfs track "best_model.pth"
   ```

2. Or upload separately (not recommended for deployment)

---

## ✅ CHECKLIST BEFORE SUBMITTING

- [ ] Local testing works (http://localhost:8000)
- [ ] Code pushed to GitHub
- [ ] Heroku app deployed and accessible
- [ ] Live URL works (https://your-app-name.herokuapp.com/)
- [ ] Model prediction works (test upload)
- [ ] Demo video recorded
- [ ] README updated with links
- [ ] All links working

---

## 📞 SUPPORT

If issues occur:

1. Check Heroku logs: `heroku logs --tail`
2. Test locally first: `python app.py`
3. Verify model file exists
4. Check requirements.txt
5. Review GitHub code

**You're done!** 🎉
