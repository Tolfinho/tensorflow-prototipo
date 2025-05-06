import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

# Configurações
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'jfif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Carrega o modelo
model = load_model("modelo_animais.h5")

# Labels (ajuste se necessário)
class_names = ['cachorro', 'gato']

# Verifica se o arquivo é permitido
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Rota principal
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocessa a imagem
            imagem = load_img(filepath, target_size=(150, 150))
            imagem_array = img_to_array(imagem) / 255.0
            imagem_array = np.expand_dims(imagem_array, axis=0)

            # Predição
            pred = model.predict(imagem_array)
            indice = np.argmax(pred)
            classe = class_names[indice]
            confianca = f"{pred[0][indice]*100:.2f}"

            return render_template("index.html", prediction=classe, confidence=confianca, image_path=filepath)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)