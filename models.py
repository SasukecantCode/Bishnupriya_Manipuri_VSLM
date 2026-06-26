import os
import json
from collections import defaultdict, Counter

class Phase1Model:
    def __init__(self):
        self.pairs = []
        
    def learn(self, en, bpm, group):
        self.pairs.append({"en": en, "bpm": bpm, "group": group})
        
    def predict(self, en, group):
        # Exact match first
        for pair in self.pairs:
            if pair["en"] == en:
                return pair["bpm"]
                
        # Group-aware retrieval (closest English by simple edit distance)
        group_pairs = [p for p in self.pairs if p["group"] == group]
        if not group_pairs:
            return ""
            
        def edit_distance(s1, s2):
            if len(s1) < len(s2):
                return edit_distance(s2, s1)
            if len(s2) == 0:
                return len(s1)
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            return previous_row[-1]
            
        best_pair = min(group_pairs, key=lambda p: edit_distance(en, p["en"]))
        return best_pair["bpm"]
        
    def confidence(self, en, group):
        for pair in self.pairs:
            if pair["en"] == en:
                return 1.0
        return 0.5 if any(p["group"] == group for p in self.pairs) else 0.0
        
    def save(self):
        pass # State is rebuilt from corpus.jsonl
        
    def load(self):
        pass

class Phase2Model:
    def __init__(self, n=3):
        self.n = n
        self.ngrams = defaultdict(lambda: defaultdict(Counter))
        self.vocab = set()
        
    def learn(self, en, bpm, group):
        padded_bpm = "^" * (self.n - 1) + bpm + "$"
        for i in range(len(padded_bpm) - self.n + 1):
            context = padded_bpm[i:i+self.n-1]
            char = padded_bpm[i+self.n-1]
            self.ngrams[group][context][char] += 1
            self.vocab.add(char)
            
    def predict(self, en, group):
        if not self.ngrams[group]:
            return ""
        
        result = ""
        context = "^" * (self.n - 1)
        
        while True:
            char_counts = self.ngrams[group].get(context, {})
            if not char_counts:
                break
            
            # Greedy decoding
            best_char = max(char_counts.items(), key=lambda x: x[1])[0]
            if best_char == "$":
                break
                
            result += best_char
            context = (context + best_char)[1:]
            
            if len(result) > 100: # limit length
                break
                
        return result
        
    def confidence(self, en, group):
        # A rough heuristic for n-gram confidence based on whether group has been seen
        return 0.7 if self.ngrams[group] else 0.0
        
    def save(self):
        pass
        
    def load(self):
        pass

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    class Phase3Network(nn.Module):
        def __init__(self, vocab_size, num_groups=21, d_model=128, nhead=4, num_layers=2):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.group_embedding = nn.Embedding(num_groups, d_model)
            self.pos_encoder = nn.Embedding(500, d_model) # max len
            
            self.transformer = nn.Transformer(
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_layers,
                num_decoder_layers=num_layers,
                batch_first=True
            )
            self.fc_out = nn.Linear(d_model, vocab_size)
            
        def forward(self, src, tgt, group_idx):
            src_pos = torch.arange(0, src.size(1), device=src.device).unsqueeze(0)
            tgt_pos = torch.arange(0, tgt.size(1), device=tgt.device).unsqueeze(0)
            
            src_emb = self.embedding(src) + self.pos_encoder(src_pos)
            
            # Inject group embedding
            grp_emb = self.group_embedding(group_idx).unsqueeze(1)
            src_emb = src_emb + grp_emb
            
            tgt_emb = self.embedding(tgt) + self.pos_encoder(tgt_pos)
            tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
            
            out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
            return self.fc_out(out)

class Phase3Wrapper:
    def __init__(self):
        if not HAS_TORCH:
            return
        self.vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.inv_vocab = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def _build_vocab(self, pairs):
        for p in pairs:
            for char in p["en"] + p["bpm"]:
                if char not in self.vocab:
                    idx = len(self.vocab)
                    self.vocab[char] = idx
                    self.inv_vocab[idx] = char
                    
    def learn(self, en, bpm, group):
        pass
        
    def _text_to_tensor(self, text):
        return torch.tensor([[self.vocab.get(c, self.vocab["<unk>"]) for c in text]], device=self.device)
        
    def predict(self, en, group):
        if not self.model or not HAS_TORCH: return ""
        self.model.eval()
        with torch.no_grad():
            src = self._text_to_tensor(en)
            grp = torch.tensor([group], device=self.device)
            tgt_indices = [self.vocab["<sos>"]]
            
            for _ in range(100):
                tgt = torch.tensor([tgt_indices], device=self.device)
                out = self.model(src, tgt, grp)
                next_token = out[0, -1].argmax().item()
                if next_token == self.vocab["<eos>"]:
                    break
                tgt_indices.append(next_token)
                
            return "".join([self.inv_vocab.get(idx, "") for idx in tgt_indices[1:]])
            
    def confidence(self, en, group):
        return 0.9 if self.model else 0.0
        
    def save(self, path="phase3_model.pt"):
        if self.model and HAS_TORCH:
            torch.save({'model_state': self.model.state_dict(), 'vocab': self.vocab}, path)
            
    def load(self, path="phase3_model.pt"):
        if HAS_TORCH and os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.vocab = checkpoint['vocab']
            self.inv_vocab = {v: k for k, v in self.vocab.items()}
            self.model = Phase3Network(len(self.vocab)).to(self.device)
            self.model.load_state_dict(checkpoint['model_state'])

class PhaseManager:
    def __init__(self):
        self.phase1 = Phase1Model()
        self.phase2 = Phase2Model()
        self.phase3 = Phase3Wrapper()
        
    def determine_phase(self, num_pairs):
        if num_pairs < 200:
            return 1
        elif num_pairs < 1000:
            return 2
        else:
            return 3 if HAS_TORCH else 2
            
    def get_model(self, num_pairs):
        phase = self.determine_phase(num_pairs)
        if phase == 1: return self.phase1
        if phase == 2: return self.phase2
        if phase == 3: return self.phase3
        
    def learn_all(self, pairs):
        for p in pairs:
            self.phase1.learn(p["en"], p["bpm"], p["group"])
            if len(pairs) >= 200:
                self.phase2.learn(p["en"], p["bpm"], p["group"])
                
    def train_phase3(self, pairs):
        if not HAS_TORCH or len(pairs) < 1000:
            return
        
        print("\n[Background] Training Phase 3 model...")
        self.phase3._build_vocab(pairs)
        self.phase3.model = Phase3Network(len(self.phase3.vocab)).to(self.phase3.device)
        optimizer = optim.Adam(self.phase3.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss(ignore_index=self.phase3.vocab["<pad>"])
        
        self.phase3.model.train()
        for epoch in range(10):
            total_loss = 0
            for p in pairs:
                src = self.phase3._text_to_tensor(p["en"])
                bpm_indices = [self.phase3.vocab["<sos>"]] + [self.phase3.vocab.get(c, self.phase3.vocab["<unk>"]) for c in p["bpm"]] + [self.phase3.vocab["<eos>"]]
                tgt = torch.tensor([bpm_indices[:-1]], device=self.phase3.device)
                tgt_y = torch.tensor([bpm_indices[1:]], device=self.phase3.device)
                grp = torch.tensor([p["group"]], device=self.phase3.device)
                
                optimizer.zero_grad()
                out = self.phase3.model(src, tgt, grp)
                loss = criterion(out.view(-1, len(self.phase3.vocab)), tgt_y.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        self.phase3.save()
        print("[Background] Phase 3 training complete!")
