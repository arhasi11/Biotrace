# 🧬 Biotrace  
### AI-Based Fingerprint Blood Group Detection System

Biotrace is an AI-powered web application that analyzes fingerprint images to predict blood groups and allows dynamic dataset expansion through user-driven registration.

---

## 🚀 Features

- 🔍 Fingerprint detection using AI (CNN model)
- 📊 Blood group prediction (A, B, AB, O)
- 🧠 Perceptual hashing for fingerprint matching
- 📝 Registration system for new fingerprints
- 📁 Automatic dataset expansion
- 🎨 Clean and modern UI (Flask + HTML/CSS)
- 🔐 Privacy-friendly (no image display)

---

## 🛠️ Tech Stack

- **Frontend:** HTML, CSS, JavaScript  
- **Backend:** Flask (Python)  
- **AI/ML:** TensorFlow / Keras (CNN Model)  
- **Image Processing:** OpenCV, PIL  
- **Hashing:** ImageHash (pHash)  

---

## 📂 Project Structure


Biotrace/
│── app.py / main.py
│── preprocessing.py
│── model/
│ └── blood_group_model.h5
│── dataset/
│ └── train/
│ ├── A/
│ ├── B/
│ ├── AB/
│ └── O/
│── static/
│ ├── css/
│ └── uploads/
│── templates/
│ ├── index.html
│ ├── result.html
│ └── register.html


---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/biotrace.git
cd biotrace
2️⃣ Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
3️⃣ Install dependencies
pip install -r requirements.txt
4️⃣ Run the application
python main.py
5️⃣ Open in browser
http://127.0.0.1:5000/
🧠 How It Works
User uploads fingerprint
System checks if it exists using perceptual hashing
If found → returns stored blood group
If not found → user can register it
New data is saved in dataset for future use
⚠️ Disclaimer

This project is for educational and research purposes only.
There is no scientifically proven relationship between fingerprints and blood groups.

📌 Future Improvements
🔐 User authentication system
📊 Accuracy improvements using pretrained models
☁️ Cloud deployment (Render / AWS)
📱 Mobile app version
📂 Database integration (MongoDB / Firebase)
👨‍💻 Author

Arhasi
B.Tech CSE Student

⭐ Contribution

Feel free to fork, improve, and contribute to this project.

📜 License

This project is licensed under the MIT License.


---

# 🔥 OPTIONAL (HIGHLY RECOMMENDED)

Also create a `requirements.txt`:

```txt
flask
tensorflow
numpy
pillow
imagehash
opencv-python
scikit-learn
