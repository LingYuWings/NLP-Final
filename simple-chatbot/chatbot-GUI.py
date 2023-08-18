import sys
import random
import json
import pickle
import numpy as np
import os

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextBrowser, QLineEdit, QPushButton, QVBoxLayout, QWidget

filename = 'intents.json'

class ChatBotGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

        self.lemmatizer = WordNetLemmatizer()
        self.intents = json.loads(open(filename).read())

        self.words = pickle.load(open('words.pkl', 'rb'))
        self.classes = pickle.load(open('classes.pkl', 'rb'))
        self.model = load_model('chatbot_model.model')

    def initUI(self):
        self.setGeometry(100, 100, 400, 600)
        self.setWindowTitle('ChatBot GUI')

        self.text_browser = QTextBrowser(self)
        self.user_input = QLineEdit(self)
        self.send_button = QPushButton('Send', self)

        layout = QVBoxLayout()
        layout.addWidget(self.text_browser)
        layout.addWidget(self.user_input)
        layout.addWidget(self.send_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.send_button.clicked.connect(self.on_send_button_click)

    def clean_up_sentence(self, sentence):
        sentence_words = nltk.wordpunct_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word) for word in sentence_words]
        return sentence_words

    def bag_of_words(self, sentence):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        for w in sentence_words:
            for i, word in enumerate(self.words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)

    def predict_class(self, sentence):
        bow = self.bag_of_words(sentence)
        res = self.model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        result = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

        result.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in result:
            return_list.append({'intent': self.classes[r[0]], 'probability': str(r[1])})
        return return_list

    def get_response(self, intents_list, intents_json):
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
        return result

    def on_send_button_click(self):
        user_message = self.user_input.text()
        self.user_input.clear()

        ints = self.predict_class(user_message)
        response = self.get_response(ints, self.intents)

        self.text_browser.append("You: " + user_message)
        self.text_browser.append("Bot: " + response)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = ChatBotGUI()
    gui.show()
    sys.exit(app.exec_())
