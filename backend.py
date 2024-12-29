from flask import Flask, request, render_template, redirect, url_for
import os
import fitz  # PyMuPDF
from PIL import Image
import sqlite3

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Database setup
DB_FILE = "questions.db"

def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                subject TEXT,
                text TEXT,
                image_path TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY,
                subject TEXT,
                category TEXT,
                text TEXT,
                image_path TEXT
            )
        """)
    conn.close()

init_db()

def process_pdf(file_path, subject):
    doc = fitz.open(file_path)
    questions = []
    for page in doc:
        text = page.get_text()
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            img_name = f"img_{subject}_{img_index}.png"
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
            with open(img_path, "wb") as img_file:
                img_file.write(img_bytes)
            questions.append({"text": text, "image": img_path})
        if not images:
            questions.append({"text": text, "image": None})
    return questions

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files["file"]
        subject = request.form.get("subject")
        if file and subject:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            questions = process_pdf(file_path, subject)
            with sqlite3.connect(DB_FILE) as conn:
                cursor = conn.cursor()
                for question in questions:
                    cursor.execute("""
                        INSERT INTO documents (subject, text, image_path)
                        VALUES (?, ?, ?)
                    """, (subject, question["text"], question["image"]))
                conn.commit()
            return redirect(url_for("view_subjects"))
    return render_template("upload.html")

@app.route("/subjects")
def view_subjects():
    return render_template("subjects.html")

@app.route("/subject/<subject>")
def view_subject(subject):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT category FROM questions WHERE subject = ?", (subject,))
        categories = cursor.fetchall()
    return render_template("subject.html", subject=subject, categories=[c[0] for c in categories])

@app.route("/category/<subject>/<category>")
def view_category(subject, category):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT text, image_path FROM questions WHERE subject = ? AND category = ?", (subject, category))
        questions = cursor.fetchall()
    return render_template("category.html", subject=subject, category=category, questions=questions)

if __name__ == "__main__":
    app.run(debug=True)
