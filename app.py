from flask import Flask, request, jsonify, send_from_directory, make_response
from Translator import load_translator, translate_text
import mysql.connector
import random
import os
from dotenv import load_dotenv
load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS")          # no default on purpose
DB_NAME = os.getenv("DB_NAME", "translator")

if DB_PASS is None:
    raise RuntimeError("DB_PASS is not set")

app = Flask(__name__)

db = mysql.connector.connect(
    host=DB_HOST,
    user=DB_USER,
    password=DB_PASS,
    database=DB_NAME
)
myCursor = db.cursor()

bundle = load_translator(ckpt_path="checkpoints/last.ckpt")

COOKIE_NAME = 'user_id'
COOKIE_MAX_AGE = 60 * 60 * 24 * 5 # 5 days

@app.route('/')
def home():
    return send_from_directory(".", "index.html")

@app.route('/translate', methods=['POST'])
def translate():
    cookie_id = request.cookies.get(COOKIE_NAME)

    if cookie_id is not None and cookie_id.isdigit():
        user_id = int(cookie_id)
        is_new_user = False
    else:
        user_id = random.randint(1_000_000_000, 9_999_999_999)
        is_new_user = True

    data = request.get_json(silent=True) or {}

    text = data.get("text", "").strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    translated_text = translate_text(bundle, text)

    if is_new_user:
        myCursor.execute(
            "INSERT IGNORE INTO Users (user_id) VALUES (%s)",
            (user_id,)
        )
        db.commit()

    myCursor.execute("INSERT INTO translations (user_id, source_text, translated_text) VALUES (%s, %s, %s)", (user_id, text, translated_text))
    db.commit()

    translation_id = myCursor.lastrowid

    res = make_response(jsonify({'translated_text': translated_text, 'translation_id': translation_id}))

    if is_new_user:
        res.set_cookie(COOKIE_NAME, 
                       str(user_id), 
                       max_age=COOKIE_MAX_AGE, 
                       httponly=True, 
                       samesite='Lax')

    return res

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json(silent=True) or {}
    user_id = request.cookies.get(COOKIE_NAME)

    if user_id is None or not user_id.isdigit():
        return jsonify({'error': "Missing user_id"}), 400
    
    user_id = int(user_id)
    
    liked = data.get("liked")
    translation_id = data.get("translation_id")

    if translation_id is None:
        return jsonify({"error": "Missing translation_id"}), 400
    try:
        translation_id = int(translation_id)
    except ValueError:
        return jsonify({"error": "translation_id must be an integer"}), 400
    
    if liked is True:
        myCursor.execute("UPDATE translations SET liked = 'Y' WHERE user_id = %s AND translation_id = %s", (user_id, translation_id))
        db.commit()
    elif liked is False:
        myCursor.execute("UPDATE translations SET liked = 'N' WHERE user_id = %s AND translation_id = %s", (user_id, translation_id))
        db.commit()
    else:
        return jsonify({"message": "No feedback provided; leaving liked as NULL"}), 200

    return jsonify({'message': 'Feedback recorded'}), 200

if __name__ == '__main__':
    app.run(debug=True)