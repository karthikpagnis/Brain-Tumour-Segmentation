# 🚀 QUICK START - ONLINE SUBMISSION (5 Steps)

Complete guide to submit your Brain Tumor Segmentation project to your teacher.

---

## ⏱️ TOTAL TIME: ~45 minutes

```
Step 1: Local Testing        → 10 min
Step 2: GitHub Push          → 5 min
Step 3: Heroku Deploy        → 15 min
Step 4: Record Video         → 10 min
Step 5: Submit to Teacher    → 5 min
```

---

## 📋 BEFORE YOU START

You need:
- ✅ `best_model.pth` (from Kaggle, 260 MB)
- ✅ `app.py` (provided in this repo)
- ✅ GitHub account (free, https://github.com)
- ✅ Heroku account (free, https://www.heroku.com)
- ✅ Git installed
- ✅ Python 3.8+

---

## STEP 1️⃣ LOCAL TESTING (10 min)

### 1.1 Download your model
From Kaggle → Outputs tab → Download `best_model.pth`

Place it in root directory:
```
Brain-Tumour-Segmentation/
├── best_model.pth   ← Add here (260 MB)
├── app.py
└── ...
```

### 1.2 Install dependencies
```bash
cd Brain-Tumour-Segmentation
pip install -r requirements.txt
```

### 1.3 Run locally
```bash
python app.py
```

### 1.4 Test in browser
Visit: `http://localhost:8000`

✅ You should see the upload interface

### 1.5 Test prediction
- Upload a test `.nii.gz` MRI file
- Click "Upload & Predict"
- Download result
- Verify it worked ✅

**If works → Continue to Step 2**

---

## STEP 2️⃣ GITHUB PUSH (5 min)

### 2.1 Create GitHub repo

Visit: https://github.com/new

- **Repository name:** Brain-Tumour-Segmentation
- **Description:** Attention-Enhanced U-Net for Brain MRI Segmentation
- **Visibility:** Public
- Click "Create repository"

### 2.2 Push your code

```bash
cd Brain-Tumour-Segmentation

git init
git add -A
git commit -m "Brain tumor segmentation API"

git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/Brain-Tumour-Segmentation.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2.3 Verify on GitHub

Visit your repo: `https://github.com/YOUR-USERNAME/Brain-Tumour-Segmentation`

You should see all files ✅

---

## STEP 3️⃣ HEROKU DEPLOY (15 min)

### 3.1 Create Heroku account
https://www.heroku.com → Sign up (free)

### 3.2 Install Heroku CLI

**macOS:**
```bash
brew tap heroku/brew && brew install heroku
```

**Windows:** 
Download from https://devcenter.heroku.com/articles/heroku-cli

**Linux:**
```bash
curl https://cli-assets.heroku.com/install.sh | sh
```

### 3.3 Login to Heroku
```bash
heroku login
```

### 3.4 Create Heroku app
```bash
heroku create your-app-name
```

Use something like: `brain-tumor-seg-2024`

**Save the URL** you get:
```
https://brain-tumor-seg-2024.herokuapp.com/
```

### 3.5 Deploy
```bash
heroku git:remote -a your-app-name
git push heroku main
```

**Wait 3-5 minutes for deployment...**

### 3.6 Check logs
```bash
heroku logs --tail
```

Should show:
```
✅ Model ready for inference
```

### 3.7 Test live app

Visit: `https://your-app-name.herokuapp.com/`

✅ Web interface should work live!

**Save this URL** - send to teacher

---

## STEP 4️⃣ RECORD DEMO VIDEO (10 min)

### 4.1 Open OBS Studio (or QuickTime/ScreenFlow)

Free download: https://obsproject.com

### 4.2 Record yourself doing this:

1. **Open the live app**
   ```
   Visit: https://your-app-name.herokuapp.com/
   ```

2. **Explain what you see**
   - "This is the Brain Tumor Segmentation API"
   - "Best Dice: 0.6189"
   - "Trained on 335 BraTS cases"

3. **Upload test MRI**
   - Drag-drop a `.nii.gz` file
   - Click "Upload & Predict"
   - Wait for result

4. **Download prediction**
   - Show download button
   - Explain what happened

5. **Show GitHub**
   - Visit GitHub repo
   - Show code structure

### 4.3 Export video

- Format: MP4 or WebM
- Duration: 5-10 minutes
- Quality: 720p or 1080p

### 4.4 Upload video

Options:
- **YouTube:** Create unlisted video
- **Google Drive:** Make shareable link
- **GitHub Releases:** Upload file

**Get the URL** - send to teacher

---

## STEP 5️⃣ SUBMIT TO TEACHER (5 min)

Send this email:

```
Subject: Brain Tumor Segmentation - Online Submission

Dear Professor/Teaching Assistant,

Please find my Brain Tumor Segmentation project submission:

📊 MODEL PERFORMANCE:
- Architecture: Attention-Enhanced U-Net 3D
- Best Dice Score: 0.6189
- Training Dataset: BraTS 2019 (335 cases)
- Epochs: 30 (best validation)

📎 SUBMISSION LINKS:

1. CODE REPOSITORY (GitHub):
   https://github.com/YOUR-USERNAME/Brain-Tumour-Segmentation

2. LIVE DEMO (Test the API):
   https://your-app-name.herokuapp.com/

3. DEMO VIDEO:
   https://youtu.be/YOUR-VIDEO-ID

📋 WHAT YOU CAN DO:

You can:
✅ Upload any 4-channel MRI file (.nii.gz)
✅ Get automatic brain tumor segmentation
✅ Download predictions as NIfTI file
✅ Review all code on GitHub
✅ Run locally: git clone... && python app.py

The model segments 4 classes:
- Background
- Necrotic Core
- Peritumoral Edema
- Enhancing Tumor

Thank you for reviewing my submission.

Best regards,
[Your Name]
```

---

## ✅ FINAL CHECKLIST

Before sending to teacher:

- [ ] Local testing works (http://localhost:8000)
- [ ] best_model.pth downloaded from Kaggle
- [ ] Code pushed to GitHub
- [ ] Heroku deployed successfully
- [ ] Live app accessible (https://your-app-name.herokuapp.com/)
- [ ] Model prediction working (tested with MRI file)
- [ ] Demo video recorded and uploaded
- [ ] All links verified and working
- [ ] Email ready to send

---

## 🆘 QUICK TROUBLESHOOTING

### "Model not loading" on Heroku
```bash
# Make sure best_model.pth is committed
git add best_model.pth
git commit -m "Add model"
git push heroku main
```

### "App crashes after deploy"
```bash
# Check logs
heroku logs --tail

# Look for error messages
```

### "Port 8000 in use (local testing)"
```bash
# Use different port
PORT=8001 python app.py
```

### "best_model.pth too large for Git"
```bash
# Use Git LFS
git lfs install
git lfs track "*.pth"
git add .gitattributes best_model.pth
git commit -m "Add model with LFS"
git push heroku main
```

---

## 🎉 YOU'RE DONE!

Your teacher can now:
1. Click the Heroku link
2. Upload an MRI file
3. See predictions in real-time
4. Review your code on GitHub
5. Watch your demo video

**Congratulations!** 🏆

---

For detailed info, see: `DEPLOYMENT.md`
