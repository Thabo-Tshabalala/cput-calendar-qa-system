//! Inference module: answers questions by searching actual document content.
//!
//! Strategy:
//!   1. Parse question into keywords
//!   2. Find keywords in raw .docx text, score windows around each hit
//!   3. Extract only relevant day-segments from the best window
//!   4. Fall back to trained Q&A pairs when confidence is high

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

fn answer_from_documents(question: &str, checkpoint: &ModelCheckpoint) -> String {
    let keywords: Vec<String> = question
        .to_lowercase()
        .split_whitespace()
        .filter(|w| w.len() > 2 && !is_stopword(w))
        .map(|w| w.to_string())
        .collect();

    let doc_result = search_documents(question, &keywords, checkpoint);
    let trained_result = retrieve_trained_answer(question, checkpoint);

    match (doc_result, trained_result) {
        (Some((file, passage, doc_score)), Some((trained_ans, qa_score))) => {
            if qa_score >= 0.40 {
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
            format!("Answer (from {}): {}\n\n[relevance: {:.2}]", file, passage, doc_score)
        }
        (None, Some((trained_ans, _))) => {
            format!("{}\n\n[Based on training data]", trained_ans)
        }
        (None, None) => {
            "No relevant information found in the calendar documents.".to_string()
        }
    }
}

/// Search raw document text: find keyword positions, score tight windows,
/// then extract only the relevant day-segments from the best window.
fn search_documents(
    question: &str,
    keywords: &[String],
    checkpoint: &ModelCheckpoint,
) -> Option<(String, String, f64)> {
    let q_lower = question.to_lowercase();
    let q_tokens = tokenize_words(&q_lower);

    if keywords.is_empty() {
        return None;
    }

    let mut best_file = String::new();
    let mut best_window = String::new();
    let mut best_score = 0.0f64;

    for doc in &checkpoint.documents {
        let content = &doc.content;
        let content_lower = content.to_lowercase();

        for kw in keywords {
            if kw.len() < 3 { continue; }
            let mut search_start = 0usize;
            while search_start < content_lower.len() {
                match content_lower[search_start..].find(kw.as_str()) {
                    None => break,
                    Some(rel_pos) => {
                        let abs_pos = search_start + rel_pos;
                        let start = snap_word_start(content, abs_pos.saturating_sub(200));
                        let end = snap_word_end(content, (abs_pos + 350).min(content.len()));
                        let window = &content[start..end];
                        let w_lower = window.to_lowercase();
                        let w_tokens = tokenize_words(&w_lower);

                        let mut score = jaccard(&q_tokens, &w_tokens);
                        let hits = keywords.iter().filter(|k| w_lower.contains(k.as_str())).count();
                        score += (hits as f64 * 0.18).min(0.72);
                        score += calendar_keyword_bonus(&q_lower, &w_lower);
                        // Prefer documents whose filename matches the year in the question
                        if q_lower.contains("2026") && doc.filename.contains("2026") { score += 0.20; }
                        if q_lower.contains("2025") && doc.filename.contains("2025") { score += 0.20; }
                        if q_lower.contains("2024") && doc.filename.contains("2024") { score += 0.20; }

                        if score > best_score {
                            best_score = score;
                            best_file = doc.filename.clone();
                            best_window = window.to_string();
                        }

                        search_start = abs_pos + 1;
                    }
                }
            }
        }
    }

    if best_score >= 0.25 && !best_window.is_empty() {
        // Extract only the day-segments that contain our keywords
        let focused = extract_relevant_segments(&best_window, keywords);
        Some((best_file, focused, best_score))
    } else {
        None
    }
}

/// Split a calendar text window into day-segments and keep only
/// the ones that contain at least one query keyword.
///
/// Calendar text looks like:
///   "... 17 Senate (12:00) 18 Research Festival Day 1 19 Research Festival Day 2 ..."
/// We split on standalone day numbers (1-31) and keep segments with keywords.
fn extract_relevant_segments(window: &str, keywords: &[String]) -> String {
    let words: Vec<&str> = window.split_whitespace().collect();
    let mut segments: Vec<Vec<&str>> = Vec::new();
    let mut current: Vec<&str> = Vec::new();

    for word in &words {
        let is_day = word.parse::<u32>().map(|n| n >= 1 && n <= 31).unwrap_or(false);
        if is_day && !current.is_empty() {
            segments.push(current.clone());
            current.clear();
        }
        current.push(word);
    }
    if !current.is_empty() {
        segments.push(current);
    }

    // Keep segments containing at least one keyword
    let relevant: Vec<String> = segments
        .iter()
        .filter(|seg| {
            let seg_text = seg.join(" ").to_lowercase();
            keywords.iter().any(|kw| seg_text.contains(kw.as_str()))
        })
        .map(|seg| seg.join(" "))
        .collect();

    if relevant.is_empty() {
        // Fall back: clean and truncate the raw window
        let cleaned = window.split_whitespace().collect::<Vec<_>>().join(" ");
        if cleaned.len() > 300 {
            format!("{}...", &cleaned[..300])
        } else {
            cleaned
        }
    } else {
        relevant.join("  |  ")
    }
}

fn snap_word_start(text: &str, pos: usize) -> usize {
    let bytes = text.as_bytes();
    let mut i = pos.min(bytes.len().saturating_sub(1));
    while i > 0 && bytes[i] != b' ' {
        i -= 1;
    }
    if i > 0 { i + 1 } else { 0 }
}

fn snap_word_end(text: &str, pos: usize) -> usize {
    let bytes = text.as_bytes();
    let mut i = pos.min(bytes.len());
    while i < bytes.len() && bytes[i] != b' ' {
        i += 1;
    }
    i
}

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

fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    let i = a.intersection(b).count();
    let u = a.union(b).count();
    if u == 0 { 0.0 } else { i as f64 / u as f64 }
}

fn calendar_keyword_bonus(q: &str, chunk: &str) -> f64 {
    let terms = [
        "graduation ceremony", "end of year graduation", "graduation", "convocation", "research festival", "open day",
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