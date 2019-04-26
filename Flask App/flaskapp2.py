from flask import Flask, request, render_template
from flask_med_luongatt_100k_testing import translate_en_fr
from googletrans import Translator


app = Flask(__name__)
def google_trans_res(text):
    translator = Translator()
    res =translator.translate(text,src='en',dest='fr')
    return res.text
@app.route("/")
def hello():
    print("Running main")
    return render_template('home.html')

@app.route("/result", methods=['POST','GET'])
def echo():
	if request.method == 'POST':
		result = request.form['text']
		g_trans = google_trans_res(result)
		real_trans,tgt_sent,bleu,lst = translate_en_fr(result)
		return render_template("output.html",result = real_trans,bleu_score=bleu,tgt_sent=tgt_sent,lst=lst,g_trans=g_trans)
    
	

if __name__ == "__main__":
    app.run(host = '0.0.0.0',port=1500)
