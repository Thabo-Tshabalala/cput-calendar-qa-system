//! Inference module: answers questions by searching actual document content.
//!
//! Strategy:
//!   1. Parse the question into keywords
//!   2. Find the keyword in the raw document text and extract surrounding context
//!   3. Score windows by keyword density + domain bonuses
//!   4. If a trained Q&A pair closely matches, show it alongside document evidence

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

/// Core QA: search documents first, trained answers second
fn answer_from_documents(question: &str, checkpoint: &ModelCheckpoint) -> String {
    let doc_result = search_documents(question, checkpoint);
    let trained_result = retrieve_trained_answer(question, checkpoint);

    match (doc_result, trained_result) {
        (Some((file, passage, doc_score)), Some((trained_ans, qa_score))) => {
            if qa_score >= 0.60 {
                format!(
                    "Answer: {}\n\nDocument Evidence [{} | relevance: {:.2}]:\n  \"{}\"",
                    trained_ans, file, doc_score, passage
                )
            } else {
                format!(
                    "Answer (from {}): {}\n\n[relevance: {:.2}]",
                    file, passage, doc_score
                )
            }
        }
        (Some((file, passage, doc_score)), None) => {
            format!(
                "Answer (from {}): {}\n\n[relevance: {:.2}]",
                file, passage, doc_score
            )
        }
        (None, Some((trained_ans, _))) => {
            format!("{}\n\n[Based on training data]", trained_ans)
        }
        (None, None) => {
            "No relevant information found in the calendar documents.".to_string()
        }
    }
}

/// PRIMARY: search raw document text for tight windows around keywords
fn search_documents(
    question: &str,
    checkpoint: &ModelCheckpoint,
) -> Option<(String, String, f64)> {
    let q_lower = question.to_lowercase();
    let q_tokens = tokenize_words(&q_lower);

    let keywords: Vec<String> = q_lower
        .split_whitespace()
        .filter(|w| w.len() > 2 && !is_stopword(w))
        .map(|w| w.to_string())
        .collect();

    if keywords.is_empty() {
        return None;
    }

    let mut best_file = String::new();
    let mut best_passage = String::new();
    let mut best_score = 0.0f64;

    for doc in &checkpoint.documents {
        let content = &doc.content;
        let content_lower = content.to_lowercase();

        // Strategy: find each keyword in the document, score a window around it
        for kw in &keywords {
            if kw.len() < 3 { continue; }
            let mut search_start = 0usize;
            while let Some(rel_pos) = content_lower[search_start..].find(kw.as_str()) {
                let abs_pos = search_start + rel_pos;

                // Extract a 400-char window centred on the keyword hit
                let start = abs_pos.saturating_sub(150);
                let end = (abs_pos + 250).min(content.len());

                // Snap to word boundaries
                let start = snap_word_start(content, start);
                let end = snap_word_end(content, end);

                let window = content[start..end].to_string();
                let w_lower = window.to_lowercase();
                let w_tokens = tokenize_words(&w_lower);

                let mut score = jaccard(&q_tokens, &w_tokens);
                let hits = keywords.iter().filter(|k| w_lower.contains(k.as_str())).count();
                score += (hits as f64 * 0.18).min(0.72);
                score += calendar_keyword_bonus(&q_lower, &w_lower);

                if score > best_score {
                    best_score = score;
                    best_file = doc.filename.clone();
                    best_passage = window.trim().to_string();
                }

                search_start = abs_pos + 1;
                if search_start >= content_lower.len() { break; }
            }
        }
    }

    if best_score >= 0.25 && !best_passage.is_empty() {
        Some((best_file, clean_passage(&best_passage, 300), best_score))
    } else {
        None
    }
}

/// Snap index to start of nearest word
fn snap_word_start(text: &str, pos: usize) -> usize {
    let bytes = text.as_bytes();
    let mut i = pos;
    while i < bytes.len() && bytes[i] != b' ' && i > 0 {
        i -= 1;
    }
    if bytes[i] == b' ' { i + 1 } else { i }
}

/// Snap index to end of nearest word
fn snap_word_end(text: &str, pos: usize) -> usize {
    let bytes = text.as_bytes();
    let mut i = pos.min(bytes.len() - 1);
    while i < bytes.len() && bytes[i] != b' ' {
        i += 1;
    }
    i.min(bytes.len())
}

/// SECONDARY: match against trained Q&A pairs
fn retrieve_trained_answer(
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
        let score = jaccard(&q_tokens, &ex_tokens) + calendar_keyword_bonus(&q_lower, &ex_lower);
        if score > best_score {
            best_score = score;
            best_answer = Some(ex.answer.clone());
        }
    }

    best_answer.map(|a| (a, best_score))
}

/// Split document into month-level chunks (fallback)
fn chunk_document(text: &str, max_words: usize) -> Vec<String> {
    let months = [
        "JANUARY","FEBRUARY","MARCH","APRIL","MAY","JUNE",
        "JULY","AUGUST","SEPTEMBER","OCTOBER","NOVEMBER","DECEMBER",
    ];

    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    let mut current_words = 0usize;
    let words: Vec<&str> = text.split_whitespace().collect();

    for word in &words {
        let is_month = months.iter().any(|m| word.eq_ignore_ascii_case(m));
        if is_month && current_words > 20 {
            if !current_chunk.trim().is_empty() {
                chunks.push(current_chunk.trim().to_string());
            }
            current_chunk = String::new();
            current_words = 0;
        }

        current_chunk.push(' ');
        current_chunk.push_str(word);
        current_words += 1;

        if current_words >= max_words {
            chunks.push(current_chunk.trim().to_string());
            let tail: Vec<String> = current_chunk
                .split_whitespace()
                .rev()
                .take(30)
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            current_words = tail.len();
            current_chunk = tail.join(" ");
        }
    }

    if !current_chunk.trim().is_empty() {
        chunks.push(current_chunk.trim().to_string());
    }

    if chunks.is_empty() { vec![text.to_string()] } else { chunks }
}

/// Clean up whitespace and truncate for display
fn clean_passage(s: &str, max_chars: usize) -> String {
    let cleaned: String = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if cleaned.len() <= max_chars {
        cleaned
    } else {
        let cut = &cleaned[..max_chars];
        match cut.rfind(' ') {
            Some(last_space) => format!("{}...", &cleaned[..last_space]),
            None => format!("{}...", cut),
        }
    }
}

fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    let i = a.intersection(b).count();
    let u = a.union(b).count();
    if u == 0 { 0.0 } else { i as f64 / u as f64 }
}

fn calendar_keyword_bonus(q: &str, chunk: &str) -> f64 {
    let terms = [
        "graduation", "convocation", "research festival", "open day",
        "hdc", "higher degrees", "senate", "council",
        "term 1", "term 2", "term 3", "term 4",
        "start of term", "end of term",
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "2024", "2025", "2026",
        "good friday", "christmas", "heritage day", "workers day",
        "youth day", "womens day", "freedom day", "reconciliation",
        "academic staff", "administrative staff", "wced",
    ];
    let bonus: f64 = terms.iter()
        .filter(|&&t| q.contains(t) && chunk.contains(t))
        .count() as f64 * 0.10;
    bonus.min(0.60_f64)
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