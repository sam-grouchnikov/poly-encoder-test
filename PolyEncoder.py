import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizerFast
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --- 1. Initialize W&B ---
wandb.init(project="polyencoder-toy-test", name="toy-polyencoder")


# --- 2. Define the Poly-Encoder model ---
class PolyEncoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", poly_m=4):
        super().__init__()
        # Base encoder: DistilBERT
        self.bert = DistilBertModel.from_pretrained(model_name)
        # Poly codes: learnable embeddings [poly_m, hidden_size]
        self.poly_codes = nn.Embedding(poly_m, self.bert.config.dim)
        self.poly_m = poly_m

    def encode_context(self, input_ids, attention_mask):
        """
        Encode context into poly-code representations.
        Args:
            input_ids: [b, T]  -- batch of tokenized context sequences
            attention_mask: [b, T]
        Returns:
            ctx_vecs: [b, m, h]  -- batch of poly-encoded context embeddings
        """
        ctx_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # ctx_out: [b, T, h]  (T = seq length, h = hidden size)

        # Expand poly codes across batch: [b, m, h]
        poly_codes = self.poly_codes.weight.unsqueeze(0).expand(ctx_out.size(0), -1, -1)

        # Attention between poly codes and context tokens: [b, m, T]
        attn = torch.bmm(poly_codes, ctx_out.transpose(1, 2))

        # Softmax over tokens
        attn_weights = torch.softmax(attn, dim=-1)

        # Weighted sum over context → [b, m, h]
        ctx_vecs = torch.bmm(attn_weights, ctx_out)
        return ctx_vecs

    def encode_candidate(self, input_ids, attention_mask):
        """
        Encode candidate sequence by taking the [CLS] token embedding
        Args:
            input_ids: [b, T]
            attention_mask: [b, T]
        Returns:
            cand_vec: [b, h]
        """
        cand_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cand_vec = cand_out[:, 0, :]  # CLS token
        return cand_vec

    def forward(self, ctx_input, ctx_mask, cand_input, cand_mask):
        """
        Compute relevance score between context and candidate.
        Returns:
            scores: [b]  -- max similarity across poly codes
        """
        ctx_vecs = self.encode_context(ctx_input, ctx_mask)  # [b, m, h]
        cand_vec = self.encode_candidate(cand_input, cand_mask)  # [b, h]

        # Dot product between each poly code and candidate: [b, m]
        scores = torch.bmm(ctx_vecs, cand_vec.unsqueeze(-1)).squeeze(-1)

        # Max over poly codes → [b]
        scores, _ = torch.max(scores, dim=-1)
        return scores


# --- 3. Toy dataset: 3 contexts, each with positive/negative candidates ---
toy_dataset = [
    ("I loved the movie!", "Me too, it was amazing!", 1),  # positive
    ("I loved the movie!", "I hated it", 0),  # negative
    ("The plot was confusing", "I agree, it was hard to follow", 1),
    ("The plot was confusing", "It was the best movie ever", 0),
    ("The acting was fantastic", "Absolutely, the cast was great", 1),
    ("The acting was fantastic", "I fell asleep", 0),
]

# --- 4. Tokenizer and model ---
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = PolyEncoder(poly_m=4).to(device)

# --- 5. Optimizer and loss ---
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
loss_fn = nn.BCEWithLogitsLoss()

# --- 6. Training loop ---
for epoch in range(10):
    total_loss = 0
    for context, candidate, label in toy_dataset:
        # Tokenize context and candidate
        ctx = tokenizer(context, return_tensors="pt", padding=True, truncation=True).to(device)
        cand = tokenizer(candidate, return_tensors="pt", padding=True, truncation=True).to(device)

        # Forward pass
        scores = model(ctx["input_ids"], ctx["attention_mask"],
                       cand["input_ids"], cand["attention_mask"])
        # scores: [b] = [1] in this toy example

        # Labels tensor
        labels = torch.tensor([label], dtype=torch.float).to(device)

        # Compute loss
        loss = loss_fn(scores, labels)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        # Log loss to W&B
        wandb.log({"train_loss": loss.item()})

    print(f"Epoch {epoch} total loss: {total_loss:.4f}")

print(model.poly_codes)
wandb.finish()
