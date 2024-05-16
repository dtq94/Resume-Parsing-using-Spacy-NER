import spacy
import random
from spacy.util import minibatch, compounding
from spacy.training.example import Example

# function to train the model
def build_spacy_model(train,model):

    config = {
                "beam_width": 1,
                "beam_density": 0.0,
                "beam_update_prob": 1.0,
                "cnn_maxout_pieces": 3,
                "nr_feature_tokens": 6,
                "nr_class": 46,
                "hidden_depth": 1,
                "token_vector_width": 96,
                "hidden_width": 64,
                "maxout_pieces": 2,
                "pretrained_vectors": None,
                "bilstm_depth": 0,
                "self_attn_depth": 0,
                "conv_depth": 4,
                "conv_window": 1,
                "embed_size": 2000
                    }
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    TRAIN_DATA =train
    
    nlp = spacy.blank('en')  # create blank Language class
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    ner = None  # Initialize ner variable
    if 'ner' not in nlp.pipe_names:
        #ner = nlp.create_pipe('ner')
        
        '''adds the Named Entity Recognition (NER) component to the spaCy pipeline. 
        The last=True argument indicates that the NER component should be added at the end of the pipeline,
        ensuring that it's the last component to process the text data.
        This allows you to focus on training only the NER component while disabling other unnecessary components.
        '''
        nlp.add_pipe('ner', last=True) 
    else:
        ner = nlp.get_pipe("ner")
        print(ner
             
             )
    # add labels
    if ner is not None:
        ner.cfg.update(config)
        for _, annotations in TRAIN_DATA:
            #print(annotations.get('entities'))
            for ent in annotations.get('entities'):
                ner.add_label(ent[2])
        # Update the NER component with the configured parameters
        
    else:
    # Handle case where ner is None
        print("NER component was not properly initialized.")
    
    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        #if model is None:
        optimizer = nlp.begin_training()
       


        for itn in range(100):
            print("Starting iteration " + str(itn))
            
            '''
            random.shuffle(TRAIN_DATA)
            losses = {}
            batches = minibatch(TRAIN_DATA, size=compounding(8., 32., 1.001))
            for batch in batches:
                 texts, annotations = zip(*batch)
                 nlp.update(texts, annotations, sgd=optimizer, 
                           losses=losses)
             print('Losses', losses)
            '''
            random.shuffle(TRAIN_DATA)
            losses = {}
            try:    
                for text, annotations in TRAIN_DATA:
                    example = Example.from_dict(nlp.make_doc(text), annotations)
                    nlp.update([example], drop=0.01, sgd=optimizer, losses=losses)
                    print('trained')
                    
            except: 
                pass
                
            print(losses)
            nlp.to_disk("model")
    return nlp
             