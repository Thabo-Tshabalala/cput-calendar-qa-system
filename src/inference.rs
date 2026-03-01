use crate::tokenizer::SimpleTokenizer;
use crate::training::ModelCheckpoint;
use std::collections::HashSet;
use std::fs;

pub fn answer_question(
    checkpoint_path: &str,
    question: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let json = fs::read_to_string(checkpoint_path)
        .map_err(|e| format!("Could not read '{}': {}", checkpoint_path, e))?;

    let checkpoint: ModelCheckpoint =
        serde_json::from_str(&json).map_err(|e| format!("Invalid checkpoint: {}", e))?;

    let _tokenizer = SimpleTokenizer::load(&checkpoint.tokenizer_path).ok();

    Ok(answer_from_documents(question, &checkpoint))
}

/// Main inference:
/// 1) Try to match a trained Q&A pair (high confidence = use directly)
/// 2) Fall back to retrieving the best document chunk
fn answer_from_documents(question: &str, checkpoint: &ModelCheckpoint) -> String {
    let qa_answer = retrieve_closest_training_answer(question, checkpoint);
    let doc_chunk = retrieve_best_doc_chunk(question, checkpoint);

    match (qa_answer, doc_chunk) {
        (Some((ans, qa_score)), Some((file, snippet, doc_score))) => {
            if qa_score >= 0.55 {
                // High confidence trained answer — return it with evidence
                format!("{}\n\n[Source: {} | confidence: {:.2}]", ans, file, doc_score)
            } else {
                // Lower confidence — show doc evidence as answer
                format!("{}\n\n[Source: {} | confidence: {:.2}]", snippet, file, doc_score)
            }
        }
        (Some((ans, _)), None) => ans,
        (None, Some((file, snippet, score))) => {
            format!("{}\n\n[Source: {} | confidence: {:.2}]", snippet, file, score)
        }
        (None, None) => {
            "I could not find a relevant answer in the calendar documents.".to_string()
        }
    }
}

/// Match question against trained Q&A pairs using Jaccard + keyword bonus
fn retrieve_closest_training_answer(
    question: &str,
    checkpoint: &ModelCheckpoint,
) -> Option<(String, f64)> {
    let q_lower = question.to_lowercase();
    let q_tokens = tokenize_words(&q_lower);

    let mut best_score = 0.0f64;
    let mut best_answer: Option<String> = None;

    for ex in &checkpoint.qa_examples {
        let ex_lower = ex.question.to_lowercase();
        let ex_tokens = tokenize_words(&ex_lower);
        let score = jaccard(&q_tokens, &ex_tokens) + keyword_bonus(&q_lower, &ex_lower);
        if score > best_score {
            best_score = score;
            best_answer = Some(ex.answer.clone());
        }
    }

    best_answer.map(|a| (a, best_score))
}

/// Retrieve best matching chunk from raw document text
fn retrieve_best_doc_chunk(
    question: &str,
    checkpoint: &ModelCheckpoint,
) -> Option<(String, String, f64)> {
    let q_lower = question.to_lowercase();
    let q_tokens = tokenize_words(&q_lower);

    let keywords: Vec<&str> = q_lower
        .split_whitespace()
        .filter(|w| w.len() > 3 && !is_stopword(w))
        .collect();

    if keywords.is_empty() {
        return None;
    }

    let mut best_file = String::new();
    let mut best_chunk = String::new();
    let mut best_score = 0.0f64;

    for doc in &checkpoint.documents {
        for chunk in split_into_chunks(&doc.content, 280) {
            let c_lower = chunk.to_lowercase();
            let c_tokens = tokenize_words(&c_lower);

            let mut score = jaccard(&q_tokens, &c_tokens);
            let hits = keywords.iter().filter(|&&kw| c_lower.contains(kw)).count();
            score += (hits as f64 * 0.10).min(0.60);
            score += keyword_bonus(&q_lower, &c_lower);

            if score > best_score {
                best_score = score;
                best_file = doc.filename.clone();
                best_chunk = chunk.trim().to_string();
            }
        }
    }

    if best_score >= 0.35 && !best_chunk.is_empty() {
        Some((best_file, shorten(&best_chunk, 220), best_score))
    } else {
        None
    }
}

/// Split document text into manageable chunks for retrieval
fn split_into_chunks(text: &str, max_len: usize) -> Vec<String> {
    let mut chunks = Vec::new();

    for para in text.split('\n') {
        let p = para.trim();
        if p.is_empty() {
            continue;
        }
        let mut current = String::new();
        for part in p.split(|c| c == '.' || c == ';' || c == ':') {
            let s = part.trim();
            if s.is_empty() {
                continue;
            }
            if current.len() + s.len() + 2 > max_len {
                if !current.trim().is_empty() {
                    chunks.push(current.trim().to_string());
                }
                current.clear();
            }
            if !current.is_empty() {
                current.push_str(". ");
            }
            current.push_str(s);
        }
        if !current.trim().is_empty() {
            chunks.push(current.trim().to_string());
        }
    }

    if chunks.is_empty() {
        vec![text.to_string()]
    } else {
        chunks
    }
}

fn shorten(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    format!("{}...", &s[..max])
}

fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    let i = a.intersection(b).count();
    let u = a.union(b).count();
    if u == 0 { 0.0 } else { i as f64 / u as f64 }
}

fn keyword_bonus(q: &str, ex: &str) -> f64 {
    let terms = [
        "graduation", "convocation", "hdc", "higher degrees",
        "term 1", "term 2", "term 3", "term 4",
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "2024", "2025", "2026", "senate", "council", "committee",
        "research", "festival", "good friday", "heritage", "christmas",
        "workers", "youth", "women", "freedom", "reconciliation",
        "wced", "academic", "administrative", "open day", "ceremony",
    ];
    let bonus: f64 = terms.iter()
        .filter(|&&t| q.contains(t) && ex.contains(t))
        .count() as f64 * 0.10;
    bonus.min(0.6_f64)
}

fn tokenize_words(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 2 && !is_stopword(w))
        .map(|w| w.to_lowercase())
        .collect()
}

fn is_stopword(word: &str) -> bool {
    matches!(
        word,
        "the" | "a" | "an" | "and" | "or" | "is" | "are" | "was" | "were"
            | "in" | "on" | "at" | "to" | "for" | "of" | "with" | "by"
            | "from" | "that" | "this" | "these" | "those" | "it" | "its"
            | "be" | "been" | "have" | "has" | "had" | "will" | "would"
            | "could" | "should" | "may" | "might" | "what" | "when"
            | "where" | "how" | "who" | "which" | "did" | "does" | "do"
    )
}