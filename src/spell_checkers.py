import os
import pickle
import random
import numpy as np
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# --- 1. Naive Bayes Corrector ---
class NaiveBayesCorrector:
    """
    Probabilistic Spell Checker (Likelihood * Prior).
    """
    def __init__(self, literature_path, alphabet=None, cache_dir='data'):
        self.words = Counter()
        self.N = 0
        self.alphabet = alphabet if alphabet else 'abcdefghijklmnopqrstuvwxyz'
        self.literature_path = literature_path
        self.cache_path = os.path.join(cache_dir, os.path.basename(literature_path) + ".nb.pkl")
        
        self.load_or_train()

    def load_or_train(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'rb') as f:
                    self.words, self.N = pickle.load(f)
                return
            except: pass
            
        self.train()

    def train(self):
        print(f"Training NaiveBayesCorrector from {self.literature_path}...")
        with open(self.literature_path, 'r', encoding='utf-8') as f:
            tokens = f.read().split()
        self.words.update(tokens)
        self.N = sum(self.words.values())
        
        # Save cache
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump((self.words, self.N), f)
        except: pass

    def P(self, word): 
        return self.words[word] / self.N

    def candidates(self, word): 
        return (self.known([word]) or self.known(self.edits1(word)) or [word])

    def known(self, words): 
        return set(w for w in words if w in self.words)

    def edits1(self, word):
        letters = self.alphabet
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def correct(self, word):
        if word in self.words: return word
        cands = self.candidates(word)
        return max(cands, key=self.P)

    def correct_text(self, text):
        return " ".join([self.correct(w) for w in text.split()])


# --- 2. KNN Corrector ---
class KNNCorrector:
    """
    KNN-based Spell Checker using Character N-Grams and Levenshtein Re-ranking.
    """
    def __init__(self, literature_path, k=1, cache_dir='data'):
        self.literature_path = literature_path
        self.k = k
        self.vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2, 4))
        self.knn = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')
        self.words_list = []
        self.fitted = False
        self.cache_path = os.path.join(cache_dir, os.path.basename(literature_path) + ".knn.pkl")
        
        self.load_or_train()

    def load_or_train(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'rb') as f:
                    data = pickle.load(f)
                    self.vectorizer = data['vectorizer']
                    self.knn = data['knn']
                    self.words_list = data['words_list']
                    self.fitted = True
                return
            except: pass
            
        self.train()

    def train(self):
        print(f"Training KNNCorrector from {self.literature_path}...")
        with open(self.literature_path, 'r', encoding='utf-8') as f:
            tokens = list(set(f.read().split()))
        self.words_list = tokens
        
        if not tokens: return

        X = self.vectorizer.fit_transform(tokens)
        self.knn.fit(X)
        self.fitted = True
        
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump({'vectorizer': self.vectorizer, 'knn': self.knn, 'words_list': self.words_list}, f)
        except: pass

    def levenshtein(self, s1, s2):
        if len(s1) < len(s2):
            return self.levenshtein(s2, s1)
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

    def correct(self, word):
        if not self.fitted: return word
        try:
            query_vec = self.vectorizer.transform([word])
            # Recommend top 50 for re-ranking
            distances, indices = self.knn.kneighbors(query_vec, n_neighbors=50)
            
            candidates = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                cand = self.words_list[idx]
                candidates.append(cand)
            
            # Re-rank by Levenshtein
            best_cand = min(candidates, key=lambda c: (self.levenshtein(word, c), len(c)))
            return best_cand
        except:
            return word

    def correct_text(self, text):
        return " ".join([self.correct(w) for w in text.split()])


# --- 3. Hybrid Corrector (KNN + Logistic Regression + Naive Bayes) ---
class HybridCorrector:
    """
    Pipeline: KNN Generation -> Logistic Regression Ranking -> Naive Bayes Scoring
    """
    def __init__(self, literature_path, cache_dir='data'):
        self.literature_path = literature_path
        self.cache_path = os.path.join(cache_dir, os.path.basename(literature_path) + ".hybrid.pkl")
        
        # Sub-models
        self.knn = KNNCorrector(literature_path, cache_dir=cache_dir)
        self.lm_words = Counter()
        self.lm_total = 0
        self.clf = LogisticRegression(class_weight='balanced')
        self.scaler = StandardScaler()
        self.fitted = False
        
        self.load_or_train()
        
    def load_or_train(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'rb') as f:
                    data = pickle.load(f)
                    self.clf = data['clf']
                    self.scaler = data['scaler']
                    self.lm_words = data['lm_words']
                    self.lm_total = data['lm_total']
                    self.fitted = True
                return
            except: pass
        
        self.train()

    def generate_typo(self, word):
        if len(word) < 2: return word
        urdu_chars = 'ابپتٹثجچحخدڈذرڑزژسشصضطظعغفقکگلمنںوہیے'
        op = random.choice(['insert', 'delete', 'replace', 'transpose'])
        word = list(word)
        idx = random.randint(0, len(word) - 1)
        if op == 'insert': word.insert(idx, random.choice(urdu_chars))
        elif op == 'delete': word.pop(idx)
        elif op == 'replace': word[idx] = random.choice(urdu_chars)
        elif op == 'transpose' and idx < len(word)-1: word[idx], word[idx+1] = word[idx+1], word[idx]
        return "".join(word)

    def extract_features(self, typo, candidate, knn_dist):
        # 1. Edit Dist
        ed = self.knn.levenshtein(typo, candidate)
        # 2. KNN Dist
        features = [ed, knn_dist, abs(len(typo) - len(candidate))]
        # 3. Start/End Match
        features.append(1 if typo and candidate and typo[0] == candidate[0] else 0)
        features.append(1 if typo and candidate and typo[-1] == candidate[-1] else 0)
        return features

    def train(self):
        print("Training HybridCorrector (this may take a minute)...")
        with open(self.literature_path, 'r', encoding='utf-8') as f:
            full_text = f.read().split()
            
        self.lm_words = Counter(full_text)
        self.lm_total = sum(self.lm_words.values())
        unique_words = list(self.lm_words.keys())
        
        # Generate LR Training Data
        X_data = []
        y_data = []
        train_samples = random.sample(unique_words, min(500, len(unique_words))) # 500 samples for speed
        
        for truth in train_samples:
            typo = self.generate_typo(truth)
            # Use KNN to find candidates
            try:
                vec = self.knn.vectorizer.transform([typo])
                dists, idxs = self.knn.knn.kneighbors(vec, n_neighbors=10)
                
                for i in range(len(idxs[0])):
                    cand = self.knn.words_list[idxs[0][i]]
                    dist = dists[0][i]
                    features = self.extract_features(typo, cand, dist)
                    label = 1 if cand == truth else 0
                    
                    X_data.append(features)
                    y_data.append(label)
            except: continue
            
        if X_data:
            X_scaled = self.scaler.fit_transform(X_data)
            self.clf.fit(X_scaled, y_data)
            self.fitted = True
            
            try:
                with open(self.cache_path, 'wb') as f:
                    pickle.dump({
                        'clf': self.clf, 
                        'scaler': self.scaler, 
                        'lm_words': self.lm_words, 
                        'lm_total': self.lm_total
                    }, f)
            except: pass

    def correct(self, word):
        if not self.fitted: return word
        if word in self.lm_words: return word # If valid, return immediately
        
        try:
            # 1. KNN Candidates
            vec = self.knn.vectorizer.transform([word])
            dists, idxs = self.knn.knn.kneighbors(vec, n_neighbors=50) # Get 50 candidates
            
            candidates = []
            feats_batch = []
            
            for i in range(len(idxs[0])):
                cand = self.knn.words_list[idxs[0][i]]
                knn_dist = dists[0][i]
                features = self.extract_features(word, cand, knn_dist)
                
                candidates.append(cand)
                feats_batch.append(features)
                
            # 2. LR Scores
            X_batch = self.scaler.transform(feats_batch)
            lr_probs = self.clf.predict_proba(X_batch)[:, 1]
            
            # 3. Naive Bayes Combination
            best_score = -1
            best_word = word
            
            for i, cand in enumerate(candidates):
                prior = self.lm_words[cand] / self.lm_total
                if prior == 0: prior = 1e-10
                
                # Final Score: LR_Likelihood * Prior
                # (Standard Bayesian Product)
                final_score = lr_probs[i] * prior
                
                if final_score > best_score:
                    best_score = final_score
                    best_word = cand
                    
            return best_word
        except:
            return word

    def correct_text(self, text):
        return " ".join([self.correct(w) for w in text.split()])
