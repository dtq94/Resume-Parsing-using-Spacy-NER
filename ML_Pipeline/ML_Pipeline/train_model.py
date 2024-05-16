import spacy
import random
from spacy.training.example import Example

def build_spacy_model(train, model=None):
    # Configuration for the NER component
    ner_config = {
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

    # Load existing model or create a blank one
    if model:
        nlp = spacy.load(model)
        print("Loaded existing model:", model)
    else:
        nlp = spacy.blank("en")
        print("Created blank 'en' model")

    # Add NER component if not already present
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")
        print("NER component already exists")

    # Update NER component with the configured parameters
    ner.cfg.update(ner_config)

    # Add entity labels
    for _, annotations in train:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # Disable other pipeline components during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        # Initialize the optimizer
        optimizer = nlp.begin_training()

        # Train the model for 100 iterations
        for itn in range(400):
            print("Starting iteration", itn)
            random.shuffle(train)
            losses = {}
            try:
                for text, annotations in train:
                    example = Example.from_dict(nlp.make_doc(text), annotations)
                    nlp.update([example], drop=0.01, sgd=optimizer, losses=losses)
                    print("Trained on example")
            except Exception as e:
                print("Error during training:", e)

            print("Losses:", losses)

    # Save the trained model to disk
    nlp.to_disk("model")

    return nlp

#https://www.youtube.com/watch?v=Md6InwA00vw