import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB



df = pd.read_csv('spam.csv')
df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
from sklearn.feature_extraction.text import CountVectorizer 

v = CountVectorizer()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df['Message'], df['spam'])
# x_train.values
x_train_c = v.fit_transform(x_train.values)

# x_train_c
# x_train_c.toarray()[:2]

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

model.fit(x_train_c, y_train)

mail = ['Enjoy the All New Dominos Pizza.','Kya aap apne mobile recharge par bachat karna chahate hain?','hello kashish']

mail_c = v.transform(mail)

model.predict(mail_c)

st.title('Spam Classifier')
st.write("This app classifies emails as spam or not spam.")

input_email = st.text_area("Enter an email message:")

if st.button("Predict"):
    if input_email:
        input_data = v.transform([input_email])
        prediction = model.predict(input_data)[0]
        if prediction == 1:
            
            st.info("This email is classified as *Spam*.")
        else:
            st.balloons()
            st.success("This email is classified as *Not Spam*.")
    else:
        st.write("Please enter an email message to classify.")