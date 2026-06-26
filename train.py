import sys
import os
import json
import re
from datetime import datetime
from collections import Counter

# --- Noun Queue & Map Handling ---
def load_noun_map():
    if os.path.exists("noun_map.json"):
        with open("noun_map.json", "r") as f:
            return json.load(f)
    return {}

def save_noun_map(nmap):
    with open("noun_map.json", "w") as f:
        json.dump(nmap, f, indent=2)

def load_noun_queue():
    if os.path.exists("noun_queue.txt"):
        with open("noun_queue.txt", "r") as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def save_noun_queue(queue):
    with open("noun_queue.txt", "w") as f:
        for noun in sorted(queue):
            f.write(f"{noun}\n")

# --- Corpus Handling ---
def load_corpus(path="corpus.jsonl"):
    pairs = []
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    pairs.append(json.loads(line))
    return pairs

def save_corpus(pairs, path="corpus.jsonl"):
    with open(path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

def extract_placeholders(bpm_text):
    return re.findall(r'\[([^\]]+)\]', bpm_text)

def parse_bank(path="bank.txt"):
    groups = []
    current = None
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith("# GROUP_"):
                parts = line.split(" ", 2)
                gid = int(parts[1].split("_")[1]) if "_" in parts[1] else int(parts[1])
                gname = parts[2] if len(parts) > 2 else parts[1]
                current = {"id": gid, "name": gname, "items": []}
                groups.append(current)
            elif current is not None:
                current["items"].append(line)
    return groups

# --- PyTorch Masked Model Reference ---
# This class demonstrates how the model trains with masked nouns
class MaskedGrammarTransformer:
    def __init__(self, vocab_size, noun_token_idx):
        self.vocab_size = vocab_size
        self.noun_token_idx = noun_token_idx
        # Setting ignore_index sets the loss contribution of <NOUN> to perfectly zero
        # during backpropagation, fulfilling the masked language model requirement.
        try:
            import torch.nn as nn
            self.criterion = nn.CrossEntropyLoss(ignore_index=self.noun_token_idx)
        except ImportError:
            pass

    def train_step(self, model, optimizer, src_encoded, tgt_encoded, tgt_expected):
        import torch
        optimizer.zero_grad()
        output = model(src_encoded, tgt_encoded)
        # Loss calculated across batch, but completely ignores any positions where target is <NOUN>
        loss = self.criterion(output.view(-1, self.vocab_size), tgt_expected.view(-1))
        loss.backward()
        optimizer.step()
        return loss.item()

# --- Two-Phase Prediction Framework ---
class SimpleGrammarModel:
    def __init__(self):
        self.pairs = []
        
    def learn(self, en, bpm_masked, group):
        self.pairs.append({"en": en, "bpm_masked": bpm_masked, "group": group})
        
    def predict_frame(self, en, group, extracted_nouns):
        group_pairs = [p for p in self.pairs if p["group"] == group]
        if not group_pairs:
            group_pairs = self.pairs
        if not group_pairs:
            return "<NOUN> VERB"
            
        def edit_distance(s1, s2):
            s1, s2 = s1.lower(), s2.lower()
            if len(s1) < len(s2): return edit_distance(s2, s1)
            if len(s2) == 0: return len(s1)
            prev = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                curr = [i + 1]
                for j, c2 in enumerate(s2):
                    ins = prev[j + 1] + 1
                    dl = curr[j] + 1
                    sub = prev[j] + (c1 != c2)
                    curr.append(min(ins, dl, sub))
                prev = curr
            return prev[-1]
            
        def get_score(p):
            c_en = p["en"].lower()
            for noun in ["river", "house", "rice", "mother", "fire"]:
                c_en = re.sub(r'\b' + noun + r'\b', '<noun>', c_en)
                
            m_en = en.lower()
            for noun in extracted_nouns:
                m_en = re.sub(r'\b' + noun + r'\b', '<noun>', m_en)
                
            # Remove noisy stop words that break up phrases
            for stop in [" the ", " a ", " an "]:
                c_en = c_en.replace(stop, " ")
                m_en = m_en.replace(stop, " ")
                
            words1 = re.findall(r'\w+', m_en)
            words2 = re.findall(r'\w+', c_en)
            
            # Unigram (single word) overlap
            uni_overlap = len(set(words1).intersection(set(words2)))
            
            # Bigram (two-word phrase) overlap
            bi1 = set(zip(words1, words1[1:]))
            bi2 = set(zip(words2, words2[1:]))
            bi_overlap = len(bi1.intersection(bi2))
            
            # Heavily weight bigram overlaps so consecutive grammar structures win out
            score = (bi_overlap * 3) + uni_overlap
            
            return (score, -edit_distance(m_en, c_en))
            
        best = max(group_pairs, key=get_score)
        return best["bpm_masked"]

class TwoPhasePredictor:
    def __init__(self):
        self.model = SimpleGrammarModel()
        self.scaffolds = ["river", "house", "rice", "mother", "fire"]
        self.corpus_vocab = set()
        
    def build_from_corpus(self, pairs):
        for p in pairs:
            for word in p["en"].lower().split():
                clean = re.sub(r'[^a-z]', '', word)
                if clean:
                    self.corpus_vocab.add(clean)
            masked = re.sub(r'\[([^\]]+)\]', '<NOUN>', p["bpm"])
            self.model.learn(p["en"], masked, p["group"])
            
    def predict(self, en, group, noun_map):
        en_nouns = []
        for word in en.lower().split():
            clean = re.sub(r'[^a-z]', '', word)
            if clean:
                if clean in self.scaffolds or clean not in self.corpus_vocab:
                    en_nouns.append(clean)
                
        frame = self.model.predict_frame(en, group, en_nouns)
        
        res = frame
        for noun in en_nouns:
            if noun in noun_map:
                res = res.replace("<NOUN>", noun_map[noun], 1)
            else:
                res = res.replace("<NOUN>", f"[{noun}]", 1)
        return res

# --- CLI Actions ---
def cmd_nouns():
    queue = load_noun_queue()
    nmap = load_noun_map()
    pairs = load_corpus()
    
    pending = [n for n in queue if n not in nmap]
    if not pending:
        print("Noun queue is empty! All nouns translated.")
        return
        
    print(f"--- Noun Training Mode ({len(pending)} pending) ---")
    for noun in pending:
        try:
            bpm_noun = input(f"Translate noun '{noun}' → ").strip()
            if bpm_noun:
                nmap[noun] = bpm_noun
                save_noun_map(nmap)
                
                # Back-fill existing pairs
                changed = 0
                for p in pairs:
                    target = f"[{noun}]"
                    if target in p["bpm"]:
                        p["bpm"] = p["bpm"].replace(target, bpm_noun)
                        if noun in p.get("placeholders", []):
                            p["placeholders"].remove(noun)
                        changed += 1
                if changed > 0:
                    save_corpus(pairs)
                    print(f"  -> Saved to map and back-filled {changed} existing pairs in corpus.")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting noun mode.")
            break
            
def cmd_stats():
    pairs = load_corpus()
    queue = load_noun_queue()
    nmap = load_noun_map()
    bank = parse_bank()
    
    print(f"Total grammar pairs: {len(pairs)}")
    print(f"Nouns translated: {len(nmap)} | Nouns in queue: {len(queue) - len(nmap)}")
    
    done_by_group = Counter(p["group"] for p in pairs)
    print("\nGroup Completion:")
    for g in bank:
        done = done_by_group.get(g["id"], 0)
        total = len(g["items"])
        print(f"  Group {g['id']} ({g['name']}): {done}/{total} ({done/max(1,total)*100:.0f}%)")
        
def cmd_fill():
    pairs = load_corpus()
    queue = load_noun_queue()
    nmap = load_noun_map()
    pending = [n for n in queue if n not in nmap]
    
    counts = Counter()
    for p in pairs:
        for ph in p.get("placeholders", []):
            if ph in pending:
                counts[ph] += 1
                
    top = counts.most_common(5)
    print("--- Top 5 Pending Nouns (Potential Impact) ---")
    total_impact = 0
    for noun, count in top:
        print(f"{noun}: would complete {count} pairs")
        total_impact += count
    print(f"\nTotal pairs that would be fully translated: {total_impact}")

def cmd_test():
    pairs = load_corpus()
    predictor = TwoPhasePredictor()
    predictor.build_from_corpus(pairs)
    nmap = load_noun_map()
    
    print("--- Test Mode (Ctrl+C to quit) ---")
    while True:
        try:
            en = input("English: ")
            if not en: break
            out = predictor.predict(en, None, nmap)
            print(f"Model Prediction: {out}\n")
        except (EOFError, KeyboardInterrupt):
            break

def cmd_review():
    pairs = load_corpus()
    if not pairs:
        print("No pairs to review.")
        return
    print("--- Review Last 20 Pairs ---")
    for p in pairs[-20:]:
        print(f"[{p['group_name']}] {p['en']} -> {p['bpm']} (Placeholders: {p.get('placeholders', [])})")

def cmd_train():
    bank = parse_bank()
    pairs = load_corpus()
    queue = load_noun_queue()
    
    done_en = {p["en"] for p in pairs}
    session = max([p.get("session", 0) for p in pairs] + [0]) + 1
    
    print(f"--- Grammar Training Mode (Session {session}) ---")
    print("Tip: Use [noun] placeholders for nouns you don't know (e.g. [river]).")
    
    added_this_session = 0
    new_placeholders = []
    
    for g in bank:
        items = [x for x in g["items"] if x not in done_en]
        for idx, item in enumerate(items):
            overall_idx = len(g["items"]) - len(items) + idx + 1
            while True:
                prompt = f"\n[Group {g['id']} · item {overall_idx}/{len(g['items'])}] Translate → {item}\n> "
                try:
                    inp = input(prompt).strip()
                except (EOFError, KeyboardInterrupt):
                    inp = "QUIT"
                    
                if inp.upper() == "QUIT":
                    print(f"\nEnding session. Added {added_this_session} pairs.")
                    
                    nmap = load_noun_map()
                    pending = [n for n in queue if n not in nmap]
                    print(f"\nNoun queue: {len(pending)} words pending translation")
                    
                    if new_placeholders:
                        counts = Counter(new_placeholders)
                        top = ", ".join([f"{k}({v})" for k, v in counts.most_common(5)])
                        print(f"Top nouns by usage frequency this session: {top}")
                    sys.exit(0)
                    
                if inp.upper() == "SKIP":
                    break
                    
                if not inp:
                    continue
                    
                # Process pair
                placeholders = extract_placeholders(inp)
                for ph in placeholders:
                    queue.add(ph)
                    new_placeholders.append(ph)
                    
                save_noun_queue(queue)
                
                pair = {
                    "en": item,
                    "bpm": inp,
                    "placeholders": placeholders,
                    "group": g["id"],
                    "group_name": g["name"],
                    "session": session,
                    "ts": datetime.now().isoformat()
                }
                pairs.append(pair)
                done_en.add(item)
                save_corpus(pairs)
                added_this_session += 1
                break

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--nouns" in args:
        cmd_nouns()
    elif "--stats" in args:
        cmd_stats()
    elif "--fill" in args:
        cmd_fill()
    elif "--test" in args:
        cmd_test()
    elif "--review" in args:
        cmd_review()
    else:
        cmd_train()
