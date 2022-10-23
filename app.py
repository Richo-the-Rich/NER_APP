import streamlit as st
from spacy import displacy
import spacy_streamlit
import spacy


def main():
    # Set the title of the app
    st.title('Named Entity Recognition App for object detection')
    # Create a menu
    menu = ['Home', 'NER']
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == 'Home':
        st.subheader('Tokenization')
        raw_text = st.text_area('Your Text', placeholder='Enter your text here')
        docx = nlp(raw_text)
        if st.button('Tokenizer'):
            spacy_streamlit.visualize_tokens(docx, attrs=['idx', 'text', 'lemma_', 'pos_', 'tag_', 'dep_', 'head', 'ent_type_'])   
    elif choice == 'NER':
        st.subheader('Named Entity Recognition')
        raw_text = st.text_area('Your Text', placeholder='Enter your text here')
        docx = nlp(raw_text)
        if st.button('NER Displacy'):
            spacy_streamlit.visualize_ner(docx)  
        
# Load the model
nlp = spacy.load("my-model")  
doc = nlp('My table is full of .bottles and chairs')
displacy.render(doc, style='ent')

if __name__ == '__main__':
    main()

   