# Deployment Guide - Traffic Intelligence Dashboard (FLOCK)

This guide provides instructions for deploying the Traffic Intelligence Dashboard to different environments.

## 1. Local Environment Setup

### Prerequisites
- Python 3.8 or higher.
- A virtual environment tool (`venv` or `conda`).

### Steps
1. **Prepare the Project:**
   Ensure you have all the necessary files:
   - `app.py`: The main Flask application.
   - `models/`: Contains `ensemble_model.pkl` and `label_encoders.pkl`.
   - `data/`: Contains `traffic_prediction_dataset.csv`.
   - `templates/`: Contains the UI files.

2. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run for Development:**
   ```bash
   python app.py
   ```
   The app will run on `http://127.0.0.1:5000`.

---

## 2. Production Deployment (General Principles)

When moving to production, do not use the built-in Flask development server (`app.run()`). Instead, use a WSGI server.

### Using Waitress (Recommended for Windows)
1. Install Waitress: `pip install waitress`
2. Run the app: `waitress-serve --port=5000 app:app`

### Using Gunicorn (Recommended for Linux/macOS)
1. Install Gunicorn: `pip install gunicorn`
2. Run the app: `gunicorn app:app`

---

## 3. Deploying to Cloud Platforms (PaaS)

### Heroku
To deploy to Heroku, you typically need to:
1.  **Create a Procfile:** This is a text file in the root directory that tells Heroku how to run your app. It should contain:
    `web: gunicorn app:app`
2.  **Login and Create:** Use the Heroku CLI (`heroku login`, `heroku create`).
3.  **Push Code:** `git push heroku main`.

### AWS / Azure / DigitalOcean (Virtual Machines)
1.  **Set up a Linux Server:** Create an EC2 instance or Droplet.
2.  **Install Python & Nginx:** Use Nginx as a reverse proxy.
3.  **Clone your repo:** Download your code onto the server.
4.  **Configure Gunicorn as a Service:** Create a systemd service file to keep the app running in the background.
5.  **Point Nginx to Gunicorn:** Configure Nginx to forward traffic from port 80 to the Gunicorn port (e.g., 8000).

---

## 4. Troubleshooting
- **Port 5000 in Use:** If `http://127.0.0.1:5000` is not opening, the port might be in use by another application. You can change the port in `app.py` at the bottom of the file: `app.run(debug=True, port=5050)`.
- **Missing Models:** The application requires the `.pkl` files in the `models/` folder. Ensure these are uploaded.
- **Path Issues:** If running on a server, ensure the working directory is set correctly so `os.path.exists()` can find the `data/` and `models/` folders.
- **Port Access:** Ensure the server's firewall (security groups) allows traffic on the port you are using (e.g., 80, 443, or 5000).
