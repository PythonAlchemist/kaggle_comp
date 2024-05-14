from transformers import AutoTokenizer, AutoModel
import torch


class SMILESToEmbedding:
    def __init__(
        self,
        tokenizer="seyonec/PubChem10M_SMILES_BPE_450k",
        model="seyonec/ChemBERTa-zinc-base-v1",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.model = AutoModel.from_pretrained(model)
        self.cache = {}

    def __call__(self, smiles: str):

        if smiles in self.cache:
            return self.cache[smiles]

        inputs = self.tokenizer(smiles, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state
        pooler_output = outputs.pooler_output

        vec = embeddings.mean(dim=1)
        self.cache[smiles] = vec
        return vec


if __name__ == "__main__":
    smiles_to_embedding = SMILESToEmbedding()
    embeddings = smiles_to_embedding("CCO")
    print("Embeddings:", embeddings)
