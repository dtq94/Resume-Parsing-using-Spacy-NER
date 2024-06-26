import spacy
import re

'''python -m spacy download en_core_web_sm''' #install this
#Function to extract names from the string using spacy
'''def extract_name(string):
    r1 = str(string)
    nlp = spacy.load('xx_ent_wiki_sm') # load the model
    doc = nlp(r1)
    for ent in doc.ents:
        if(ent.label_ == 'PER'):
            print(ent.text)
            break'''
#Function to extract Phone Numbers from string using regular expressions
def extract_phone_numbers(string):
    r = re.compile(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})')
    phone_numbers = r.findall(string)
    return [re.sub(r'\D', '', number) for number in phone_numbers]


#Function to extract Email address from a string using regular expressions
def extract_email_addresses(string):
    r = re.compile(r'[\w\.-]+@[\w\.-]+')
    return r.findall(string)

def extract_name(string):
    nlp = spacy.load('en_core_web_sm')  # Load the English NER model
    doc = nlp(string)
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            return ent.text
            # Remove the break statement if you want to extract all person entities
            break