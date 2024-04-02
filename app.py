#<<<×>>>


import tkinter
from tkinter import messagebox 
import os
#import sys
#sys.path.append("/stat/")

import tkinter as tk
from transformers import BertTokenizer, TFDistilBertModel
import tensorflow as tf
import numpy as np
import transformers
import tensorflow as tf
from keras.layers import Dense, Input, Dropout, Lambda
from keras.optimizers import Adam


output_path = "outputs"
    
try:
    tokenizer = transformers.AutoTokenizer.from_pretrained(output_path+'/tokenizers')
except (OSError, ValueError):
    
    tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
    # Save the loaded tokenizer locally
    tokenizer.save_pretrained(output_path+'/tokenizers')
    print("save tokenizer")


def build_bert_model(max_len=192, optimizer=Adam(learning_rate=1e-5)):
  """
  That function create the BERT model for training
  """
  # Charger le modèle pré-entraîné DistilBERT et le tokenizer
  distilbert_model = TFDistilBertModel.from_pretrained('distilbert-base-multilingual-cased')
  #tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')

  model = tf.keras.Sequential([
    # La couche d'entrée
    Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids"),

    # Ajouter la couche DistilBERT (notez que nous utilisons distilbert_model.layers[0] pour accéder à la couche de transformer)
    # La couche DistilBERT
    distilbert_model.layers[0],

    # La couche pour obtenir le premier token [CLS]
    Lambda(lambda seq: seq[:, 0, :]),

    # La couche de sortie
    Dense(1, activation='sigmoid')
  ])

  loss = tf.keras.losses.BinaryCrossentropy()
  #metrics = tf.metrics.BinaryAccuracy()

  # Compiler le modèle
  # Compiler le modèle avec une loss adaptée à la classification binaire
  model.compile(optimizer = optimizer, loss=loss, metrics=['accuracy'])

  return model

new_model = build_bert_model()  # Créez le modèle avec la même architecture
new_model.load_weights(output_path+'/trained-models/bert_model-3-val.h5') 

# Fonction pour prédire si la phrase est toxique
def predict_toxicity():
    # Obtenir la phrase saisie par l'utilisateur
    phrase = entry.get()

    # Prétraiter la phrase
    inputs = tokenizer(phrase, return_tensors="tf", max_length=192, truncation=True, padding='max_length')

    # Obtenir la prédiction
    predictions = new_model.predict(inputs['input_ids'])[0]

    # Interpréter la prédiction
    toxic_threshold = 0.5
    is_toxic = predictions[0] > toxic_threshold
    
    per = predictions[0] * 100

    print(f" **{phrase}** a une toxicité de {per:.2f} - [{'toxique' if is_toxic else 'non-toxique'}]")

    # Afficher le résultat de la prédiction
    if is_toxic:
        result_label.config(text=f"Toxicité de [{per:.2f}%] | classé : Toxique", fg="red")
    else:
        result_label.config(text=f"Toxicité de [{per:.2f}%] | classé : Non-toxique", fg="green")


#***Foction Manage***

def presentation(screen):
        # Vernada
        f2 = ("Helvetica",-14,)
        presentationFrame = tkinter.LabelFrame(screen)
        presentationFrame.pack()#grid(row=0,column=0,columnspan=5, pady=(5,5), padx=(15,15))
        tkinter.Label(presentationFrame,text="IFT714", font = f2).grid(row=0,column=0)
        #tkinter.Label(presentationFrame,text="...", font = f2).grid(row=1,column=0)
        tkinter.Label(presentationFrame,text="Par Équipe 4", font = f2).grid(row=2,column=0)
        
        ptf = tkinter.LabelFrame(presentationFrame)
        ptf.config(bg="blue")
        ptf.grid(row=3,column=0,padx=(10,10), pady=(0,5))
        tkinter.Label(ptf,text="Outil de detection de toxicité").grid(row=1,column=0)

def about_app():
    about_w = tkinter.Toplevel(mainScreen)
    about_w.title("A propos")
    about_w.minsize(260,140)
    f2 = ("Helvetica",-14,)
    presentationFrame = tkinter.LabelFrame(about_w)
    presentationFrame.pack(padx=15,pady=15)#grid(row=0,column=0,columnspan=5, pady=(5,5), padx=(15,15))
    tkinter.Label(presentationFrame,
    text="Fait par : Hassan Tshelaka Kajila \n" 
        + "Kevin Vaneck Nana \n" 
        + "Fatou Dia \n", 
    font = f2).grid(row=0,column=0)
    tkinter.Label(presentationFrame,text="IFT714 - TALN", font = f2).grid(row=1,column=0)
    tkinter.Label(presentationFrame,text="Dirigé par : Prof. Amine Trabelesi", font = f2).grid(row=2,column=0)
    ptf = tkinter.LabelFrame(presentationFrame)
    ptf.config(bg="blue")
    ptf.grid(row=3,column=0,padx=(10,10), pady=(0,5))
    tkinter.Label(ptf,text="Outil de detection de toxicité").grid(row=1,column=0)


def messagetoquit():
    ask_ = messagebox.askokcancel("Confirmation","Voulez-vous vraiment quitter?")
    if ask_:
        window.quit()
   

    else:
        print("_")


def messagetonotinfo():
    messagebox.showinfo("Pas d'info","C'est ne pas fonctionnel pour l'intant")


def fs(change = True):
    window.attributes("-fullscreen", not window.attributes("-fullscreen"))
    print(type(change))
    print(change)
    
    if(change):
        menu3.delete(6)
        menu3.insert_cascade(6,label="Normal Screen [Esc]",command = lambda: fs(False))
        window.bind('<Escape>', lambda e: fs(False))
    else:
        menu3.delete(6)
        menu3.insert_cascade(6,label="Full Screen",command = lambda: fs())
        window.unbind('<Escape>', lambda e: fs(False))

def open_guide():
    try:
        from tkPDFViewer import tkPDFViewer as pdf
  
        # Initializing tk
        pdfView = tkinter.Toplevel()
        pdfView.title("Manuel d'utilisation")
        pdfView.iconbitmap("image.ico")
        pdfView.geometry("620x750")
        v1 = pdf.ShowPdf()
        v2 = v1.pdf_view(pdfView,pdf_location = r"Guide.pdf", width = 180, height = 100)
        
        
        v2.pack()
    except Exception as e:
        print("#run time error : ", e)
        os.system("final-report.pdf")

# ########################### Parametre de fnt ################################
window = tkinter.Tk()
window.title("NLP")
window.minsize(480,360)
window.geometry("680x360")
fontHelvetica = ("Helvetica",-14,"bold")
fontCourierNew = ("Courier New",-14,"bold")

#window.config(background="")
#window.iconbitmap("image.ico")

#window.bind('<space>', lambda e: about_app())


#photo = tkinter.PhotoImage(file="image.png" )
#panneau= tkinter.Label(window, image= photo)
#panneau.config(bg='systemTransparent')
#panneau.place(x="15", y="15")
mainScreen = tkinter.Frame(window)
mainScreen.pack()
#***Widgets***
mainmenu = tkinter.Menu(window)

menu0 = tkinter.Menu(mainmenu, tearoff=0)
menu0.add_command(label="Nouveau Fichier")
menu0.add_command(label="Ouvrir",command= lambda: about_app())
menu0.add_separator()
#menu0.add_command(label="______________")
menu0.add_command(label="Enregistrer")
menu0.add_command(label="Print")
menu0.add_separator()
#menu0.add_command(label="______________")
menu0.add_command(label="Close")
menu0.add_command(label="Exit",command=messagetoquit)

menu1 = tkinter.Menu(mainmenu, tearoff=1)
menu1.add_command(label="Stat Univarié",)
menu1.add_command(label="Stat Bivariée",)
menu1.add_separator()
menu1.add_command(label="Dist de probabilité",)


menu2 = tkinter.Menu(mainmenu, tearoff=0)
menu2.add_command(label="Manuel d'utilisation", command = open_guide)
menu2.add_command(label="A propos", command=about_app)

menu3 = tkinter.Menu(mainmenu,tearoff=0)
menu3.add_command(label="Format ")
menu3.add_separator()
menu3.add_command(label="480x360",command= lambda: window.geometry("480x360"))
menu3.add_command(label="640x480",command= lambda: window.geometry("640x480"))
menu3.add_command(label="820x560",command= lambda: window.geometry("820x560"))

menu3.add_separator()
menu3.add_command(label="Full Screen", command = lambda: fs())

menuz = tkinter.Menu(mainmenu, tearoff=0)
menuz.add_command(label="About",command=about_app)


mainmenu.add_cascade(labe="Fichier",menu=menu0)
mainmenu.add_cascade(label="Data",menu=menu1)
mainmenu.add_cascade(label="Format",menu=menu3)
#mainmenu.add_cascade(label="Info ",menu=menuz)
mainmenu.add_cascade(label="Info",menu=menu2)


mainframe0 = tkinter.Frame(mainScreen)
mainframe1 = tkinter.LabelFrame(mainScreen,text="Traitement",borderwidth=2)
mainframe2 = tkinter.Frame(mainScreen,width=180, height=180,borderwidth=5)

#rool = tkinter.Scale(mainScreen)
btn1 = tkinter.Button(mainframe1, text="Prédire",borderwidth=5, highlightthickness = 3, width=30,command= lambda: predict_toxicity())
btn2 = tkinter.Button(mainframe1, text="Statistique bivariée",borderwidth=4, highlightthickness = 3, width=30,command= lambda: None)
btn3 = tkinter.Button(mainframe1, text="Distribution de probabilité", borderwidth=4, highlightthickness = 3, width=30,command= lambda: None)

mainframe0.pack(pady=5)#.grid(row=1,column=1,columnspan=2)
mainframe1.pack()#.grid(row=2,column=0)
mainframe2.pack()#.grid(row=2,column=1)
presentation(mainframe0)

#       Frame1
tkinter.Label(mainframe1,text = " ---    Saisir une phrase    --- ").pack(pady=(5,0))
entry = tk.Entry(mainframe1, width=75, font=("Arial", 12, "bold"))
#entry.config(height=16)
entry.pack(pady=(5,5), padx=(16,16))
btn1.pack(pady=(5,20),padx=15)
#btn2.pack(pady=2, padx=15)
tkinter.Label(mainframe1,text = "---    Résultat    --- ").pack()
#btn3.pack(pady=2, padx=15)

result_label = tkinter.Label(mainframe1, text="Aucun résultat", font=("Arial", 12, "bold"))
result_label.pack()



#***Loop***
window.config(menu=mainmenu)
window.protocol("WM_DELETE_WINDOW",messagetoquit)
window.mainloop()




