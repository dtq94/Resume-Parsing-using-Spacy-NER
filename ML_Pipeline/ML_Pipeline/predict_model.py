import spacy
import re
from ML_Pipeline import text_extractor
from ML_Pipeline import entity_extractor

# function for prediction
def predict(path):
    output_list = []
    nlp=spacy.load('model') # load the model
    test_text=text_extractor.convert_pdf_to_text(path) # convert
        
    for text in test_text:
        output={}
        name = entity_extractor.extract_name(text)
        
        #ph_no = entity_extractor.extract_phone_numbers(text)
        #email = entity_extractor.extract_email_addresses(text)

        text=text.replace('\n',' ') # replace
        doc = nlp(text)
        
        
        output['Name']= name
        
        
        for ent in doc.ents:
            #print(f'{ent.label_.upper():{30}}-{ent.text}')
            #print(ent.text, ent.label_.upper())
            output[ent.label_.upper()]=ent.text
        output_list.append(output)  # Add the output dictionary to the list
    
    #return output

    print(output_list)
    