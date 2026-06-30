import json
import os
from train import parse_bank, load_corpus
from models import PhaseManager

def evaluate():
    groups = parse_bank()
    pairs = load_corpus()
    
    if not groups:
        print(f"Error: Bank data not found.")
        return
        
    print("=" * 50)
    print("BISHNUPRIYA MANIPURI VSLM - EVALUATION REPORT")
    print("=" * 50)
    
    # A. Group-level coverage score
    print("\n--- A. Group-Level Coverage Score ---")
    done_by_group = {g["id"]: set() for g in groups}
    for p in pairs:
        done_by_group[p["group"]].add(p["en"])
        
    completed_groups = []
    
    for g in groups:
        total = len(g["items"])
        done = len(done_by_group[g["id"]])
        coverage = (done / max(1, total)) * 100
        print(f"Group {g['id']} ({g['name']}): {coverage:.1f}% ({done}/{total})")
        if coverage == 100.0:
            completed_groups.append(g)
            
    # B. Grammar Generalisation Test
    print("\n--- B. Grammar Generalisation Test ---")
    print("Testing morpheme generalisation on completed groups (Hold-out method)...")
    
    if not completed_groups:
        print("No groups are 100% completed yet. Complete a group to run generalisation test.")
    else:
        manager = PhaseManager()
        # For evaluation, we hold out 2 items from each completed group.
        # We train on the rest, and evaluate on the 2 held out.
        for g in completed_groups:
            if len(g["items"]) < 3:
                continue
            # Pick first 2 as held-out (deterministic for evaluation)
            held_out = g["items"][:2]
            train_items = set(g["items"][2:])
            
            train_pairs = [p for p in pairs if p["en"] in train_items or p["group"] != g["id"]]
            test_pairs = [p for p in pairs if p["en"] in held_out]
            
            # Train temp model
            manager.learn_all(train_pairs)
            model = manager.get_model(len(train_pairs))
            
            print(f"\nGroup {g['id']} ({g['name']}):")
            for tp in test_pairs:
                en = tp["en"]
                actual = tp["bpm"]
                pred = model.predict(en, g["id"])
                
                # Check for morpheme match roughly (e.g. at least 3 character overlap in suffix, or just Levenshtein)
                # Here we do a simple exact string check or print it for human evaluation.
                match = (pred == actual)
                print(f"  EN: {en}")
                print(f"  Expected: {actual}")
                print(f"  Predicted: {pred}")
                print(f"  Transfer Successful: {'Yes' if match else 'No (needs more Phase 2/3 data)'}")

    # C. Paradigm Completion Test (Group 2 specific)
    print("\n--- C. Paradigm Completion Test ---")
    group2 = next((g for g in groups if g["id"] == 2), None)
    if group2:
        total_pronouns = len(group2["items"])
        done = len(done_by_group[2])
        print(f"Pronoun Paradigm Completion: {done}/{total_pronouns} ({(done/max(1, total_pronouns))*100:.1f}%)")
        if done < total_pronouns and done > 0:
            print("Model is partially trained on pronouns. Testing completion...")
            # We could ask the model to predict the missing ones
            manager_full = PhaseManager()
            manager_full.learn_all(pairs)
            model_full = manager_full.get_model(len(pairs))
            missing = [item for item in group2["items"] if item not in done_by_group[2]]
            for m in missing:
                pred = model_full.predict(m, 2)
                print(f"  Missing: {m}")
                print(f"  Model Guess: {pred if pred else '(No guess)'}")
    else:
        print("Group 2 (Pronouns) not found in bank.")

if __name__ == "__main__":
    evaluate()
