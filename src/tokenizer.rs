use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;

/// Special tokens
pub const PAD_TOKEN: &str = "[PAD]";
pub const UNK_TOKEN: &str = "[UNK]";
pub const CLS_TOKEN: &str = "[CLS]";
pub const SEP_TOKEN: &str = "[SEP]";
pub const MASK_TOKEN: &str = "[MASK]";

/// A simple word-level tokenizer that builds vocabulary from training data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleTokenizer {
    pub vocab: HashMap<String, u32>,
    pub id_to_token: HashMap<u32, String>,
    pub vocab_size: usize,
    pub max_seq_len: usize,
}

impl SimpleTokenizer {
    /// Create a new tokenizer with special tokens pre-populated
    pub fn new(max_seq_len: usize) -> Self {
        let mut vocab = HashMap::new();
        let mut id_to_token = HashMap::new();

        // Reserve special tokens at the start
        for (i, tok) in [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN].iter().enumerate() {
            vocab.insert(tok.to_string(), i as u32);
            id_to_token.insert(i as u32, tok.to_string());
        }

        let vocab_size = vocab.len();

        Self {
            vocab,
            id_to_token,
            vocab_size,
            max_seq_len,
        }
    }

    /// Build vocabulary from a collection of texts
    pub fn build_vocab(&mut self, texts: &[&str]) {
        let mut word_freq: HashMap<String, usize> = HashMap::new();

        for text in texts {
            for token in self.tokenize_to_words(text) {
                *word_freq.entry(token).or_insert(0) += 1;
            }
        }

        // Sort by frequency (descending) for consistent vocab
        let mut words: Vec<(String, usize)> = word_freq.into_iter().collect();
        words.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

        for (word, _freq) in words {
            if !self.vocab.contains_key(&word) {
                let id = self.vocab.len() as u32;
                self.id_to_token.insert(id, word.clone());
                self.vocab.insert(word, id);
            }
        }

        self.vocab_size = self.vocab.len();
        println!("  Vocabulary size: {}", self.vocab_size);
    }

    /// Tokenize text into word tokens (lowercase, simple punctuation split)
    fn tokenize_to_words(&self, text: &str) -> Vec<String> {
        let text = text.to_lowercase();
        let mut tokens = Vec::new();
        let mut current = String::new();

        for ch in text.chars() {
            if ch.is_alphanumeric() || ch == '\'' {
                current.push(ch);
            } else {
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
                if !ch.is_whitespace() && !ch.is_control() {
                    // Keep punctuation as tokens
                    tokens.push(ch.to_string());
                }
            }
        }
        if !current.is_empty() {
            tokens.push(current);
        }
        tokens
    }

    /// Encode text to token IDs, with optional padding/truncation
    pub fn encode(&self, text: &str, max_len: Option<usize>) -> Vec<u32> {
        let max_len = max_len.unwrap_or(self.max_seq_len);
        let unk_id = *self.vocab.get(UNK_TOKEN).unwrap_or(&1);

        let mut ids: Vec<u32> = self
            .tokenize_to_words(text)
            .iter()
            .map(|tok| *self.vocab.get(tok).unwrap_or(&unk_id))
            .take(max_len - 2) // Leave room for CLS/SEP
            .collect();

        // Add CLS at start, SEP at end
        let cls_id = *self.vocab.get(CLS_TOKEN).unwrap_or(&2);
        let sep_id = *self.vocab.get(SEP_TOKEN).unwrap_or(&3);
        let pad_id = *self.vocab.get(PAD_TOKEN).unwrap_or(&0);

        ids.insert(0, cls_id);
        ids.push(sep_id);

        // Pad to max_len
        while ids.len() < max_len {
            ids.push(pad_id);
        }

        ids
    }

    /// Encode a context + question pair (separated by SEP)
    pub fn encode_qa(&self, context: &str, question: &str, max_len: Option<usize>) -> Vec<u32> {
        let max_len = max_len.unwrap_or(self.max_seq_len);
        let unk_id = *self.vocab.get(UNK_TOKEN).unwrap_or(&1);
        let cls_id = *self.vocab.get(CLS_TOKEN).unwrap_or(&2);
        let sep_id = *self.vocab.get(SEP_TOKEN).unwrap_or(&3);
        let pad_id = *self.vocab.get(PAD_TOKEN).unwrap_or(&0);

        let ctx_tokens: Vec<u32> = self
            .tokenize_to_words(context)
            .iter()
            .map(|tok| *self.vocab.get(tok).unwrap_or(&unk_id))
            .collect();

        let q_tokens: Vec<u32> = self
            .tokenize_to_words(question)
            .iter()
            .map(|tok| *self.vocab.get(tok).unwrap_or(&unk_id))
            .collect();

        // [CLS] ctx_tokens [SEP] q_tokens [SEP]
        let total_needed = 3 + ctx_tokens.len() + q_tokens.len();
        let mut ids = vec![cls_id];

        if total_needed <= max_len {
            ids.extend_from_slice(&ctx_tokens);
            ids.push(sep_id);
            ids.extend_from_slice(&q_tokens);
            ids.push(sep_id);
        } else {
            // Truncate context to fit
            let q_space = q_tokens.len().min(max_len / 3);
            let ctx_space = max_len - q_space - 3;
            ids.extend_from_slice(&ctx_tokens[..ctx_tokens.len().min(ctx_space)]);
            ids.push(sep_id);
            ids.extend_from_slice(&q_tokens[..q_tokens.len().min(q_space)]);
            ids.push(sep_id);
        }

        // Pad
        while ids.len() < max_len {
            ids.push(pad_id);
        }

        ids[..max_len].to_vec()
    }

    /// Decode token IDs back to text
    pub fn decode(&self, ids: &[u32]) -> String {
        let pad_id = *self.vocab.get(PAD_TOKEN).unwrap_or(&0);
        let special: std::collections::HashSet<u32> = vec![
            *self.vocab.get(CLS_TOKEN).unwrap_or(&2),
            *self.vocab.get(SEP_TOKEN).unwrap_or(&3),
            pad_id,
        ].into_iter().collect();

        ids.iter()
            .filter(|id| !special.contains(id))
            .map(|id| self.id_to_token.get(id).cloned().unwrap_or_else(|| "[UNK]".to_string()))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Save tokenizer to a JSON file
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Load tokenizer from a JSON file
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let json = fs::read_to_string(path)?;
        let tokenizer: Self = serde_json::from_str(&json)?;
        Ok(tokenizer)
    }
}
