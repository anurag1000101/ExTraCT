from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


class Extract:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def generate_bert_embeddings(self, textual_description: list):
        encoded_input = self.tokenizer(
            textual_description, padding=True, truncation=True, return_tensors="pt"
        )
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling
        sentence_embeddings = self.mean_pooling(
            model_output, encoded_input["attention_mask"]
        )

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings

    def generate_features(self, objects, templates):
        features = []
        keys = []
        names = []

        for key in templates.keys():
            if "obj" in key:
                for objs in objects:
                    name = objs["name"]
                    for TFT in templates[key]:
                        features.append(TFT.format(obj=name))
                        keys.append(key)
                        names.append(name)

            else:
                for TFT in templates[key]:
                    features.append(TFT)
                    keys.append(key)
                    names.append("no_object")
        return {"TDT": features, "FT": keys, "objects": names}

    def compute_cosine_similarity(self, vector_1, vector_2):
        return F.cosine_similarity(vector_1, vector_2, dim=1)
