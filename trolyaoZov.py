import os
import speech_recognition
import io
import random
import string # to process standard python strings
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import pyttsx3
import pywinauto
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True)
warnings.filterwarnings('ignore')
# setting AI listens

# set ngon ngu phan AI noi
language = 'vi'
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty("voice", voices[1].id)
engine.setProperty("rate", 150)
noi=""
# Xu ly data
nghe=speech_recognition.Recognizer()
def lis():
    with speech_recognition.Microphone() as mic:
        print("Tôi dang nghe")
        audio = nghe.record(mic,duration=5)
        try:
            you = nghe.recognize_google(audio, language='vi-VN')
            print("you: "+you)
            return(you)
        except:
            print("...")
#nltk.download('punkt') # first-time use only
#nltk.download('wordnet') # first-time use only

with open('khodata.txt','r', encoding='utf8', errors ='ignore') as fin:
    raw = fin.read().lower()
#TOkenisation
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words

# Preprocessing
lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

GREETING_INPUTS = ("xin chào", "hi", "chào", "ê", "a lô","hê lô")
GREETING_RESPONSES = ["chào bạn", "Tôi là Zov, hân hạnh được trò chuyện với bạn"]


def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:       	
            return (random.choice(GREETING_RESPONSES))

def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"tôi xin lỗi, tôi không hiểu"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response
    
def speak(text):
    print("Bot: {}".format(text))
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

flag=True
# Phan AI noi
print("Tên tôi là Zov, chúng ta hãy bắt đầu nào!")
print("Danh sách ngôn ngữ đáng học tập:\n15, TypeScript\n14, Swift\n13, Scala\n12, Objective-C\n11, Shell\n10, Go\n9, C\n8, C#\n7, CSS\n6,C++ \n5, PHP\n4, Ruby\n3, Python\n2, Java\n1, JavaScript")
while(flag==True):
    user_response = lis()
    #user_response=user_response.lower()
    if(user_response!='tạm biệt'):
        if(user_response=='cám ơn' or user_response=='cám ơn bạn' ):
            flag=False
            noi=("Bạn luôn được chào đón..")
            #print(noi)
        else:      
            if((user_response)==None):
            	#noi=(greeting(user_response))
            	#print(noi)
            	#engine.say(noi)
                flag=False
                noi=("Tạm biệt, bảo trọng nhé..!")              
            elif(greeting(user_response)!=None):
                noi=(greeting(user_response))
            else:
                print(end="")
                noi=(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        noi=("Tạm biệt, bảo trọng nhé..!")

    speak(noi)

#
