//! Inference module: answers questions by searching actual document content.
//!
//! Strategy:
//!   1. Parse the question into keywords
//!   2. Search all loaded .docx document chunks for the best matching passage
//!   3. If a very close trained Q&A pair exists (jaccard >= 0.6), show it alongside
//!   4. Always ground the answer in actual document text

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

/// Core QA logic — always fetches answer from actual document text
fn answer_from_documents(question: &str, checkpoint: &ModelCheckpoint) -> String {
    // Step 1: Search the actual .docx document content
    let doc_result = search_documents(question, checkpoint);

    // Step 2: Check if a trained Q&A pair closely matches (confidence >= 0.6)
    let trained_result = retrieve_trained_answer(question, checkpoint);

    match (doc_result, trained_result) {
        // Both document and trained answer found
        (Some((file, passage, doc_score)), Some((trained_ans, qa_score))) => {
            if qa_score >= 0.60 {
                // High-confidence trained answer — show it, backed by document evidence
                format!(
                    "Answer: {}\n\nDocument Evidence [{} | relevance: {:.2}]:\n  \"{}\"",
                    trained_ans, file, doc_score, passage
                )
            } else {
                // Lower confidence — show document passage as primary answer
                format!(
                    "Answer (from documents): {}\n\nSource: {} | relevance: {:.2}",
                    passage, file, doc_score
                )
            }
        }
        // Only document found
        (Some((file, passage, doc_score)), None) => {
            format!(
                "Answer (from documents): {}\n\nSource: {} | relevance: {:.2}",
                passage, file, doc_score
            )
        }
        // Only trained answer found (fallback)
        (None, Some((trained_ans, _))) => {
            format!("Answer: {}\n\n[Based on training data — no direct document match found]", trained_ans)
        }
        // Nothing found
        (None, None) => {
            "No relevant information found in the calendar documents for that question.".to_string()
        }
    }
}

/// Search actual document text for the best matching passage.
/// This is the PRIMARY answer source — real document content.
fn search_documents(
    question: &str,
    checkpoint: &ModelCheckpoint,
) -> Option<(String, String, f64)> {
    let q_lower = question.to_lowercase();
    let q_tokens = tokenize_words(&q_lower);

    // Extract meaningful keywords from question
    let keywords: Vec<&str> = q_lower
        .split_whitespace()
        .filter(|w| w.len() > 2 && !is_stopword(w))
        .collect();

    if keywords.is_empty() {
        return None;
    }

    let mut best_file = String::new();
    let mut best_passage = String::new();
    let mut best_score = 0.0f64;

    for doc in &checkpoint.documents {
        // Split document into sentence-level chunks for precise retrieval
        for chunk in chunk_document(&doc.content, 300) {
            let c_lower = chunk.to_lowercase();
            let c_tokens = tokenize_words(&c_lower);

            // Base similarity score
            let mut score = jaccard(&q_tokens, &c_tokens);

            // Bonus for keyword hits in this chunk
            let hits = keywords
                .iter()
                .filter(|&&kw| c_lower.contains(kw))
                .count();
            score += (hits as f64 * 0.12).min(0.65);

            // Domain-specific bonuses for calendar terms
            score += calendar_keyword_bonus(&q_lower, &c_lower);

            if score > best_score {
                best_score = score;
                best_file = doc.filename.clone();
                best_passage = chunk.trim().to_string();
            }
        }
    }

    // Only return if we found something meaningful
    if best_score >= 0.30 && !best_passage.is_empty() {
        // Clean up and shorten the passage for readable output
        let clean = clean_passage(&best_passage, 250);
        Some((best_file, clean, best_score))
    } else {
        None
    }
}

/// Retrieve the closest trained Q&A answer (secondary source)
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

/// Split document content into overlapping chunks for better retrieval coverage.
/// Calendar text is space-separated events, so we split on whitespace boundaries.
fn chunk_document(text: &str, max_words: usize) -> Vec<String> {
    let mut chunks = Vec::new();

    // First try splitting on month headings (e.g. "JANUARY 2026", "FEBRUARY 2024")
    // which naturally segment the calendar into monthly sections
    let months = [
        "JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE",
        "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER",
    ];

    let mut current_chunk = String::new();
    let mut current_words = 0usize;
    let words: Vec<&str> = text.split_whitespace().collect();

    let mut i = 0;
    while i < words.len() {
        let word = words[i];

        // Detect month boundary — start new chunk
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

        // Also split on max length
        if current_words >= max_words {
            chunks.push(current_chunk.trim().to_string());
            // Overlap: keep last 30 words for context continuity
            let overlap_start = current_words.saturating_sub(30);
            let overlap_words: Vec<String> = current_chunk
                .split_whitespace()
                .skip(overlap_start)
                .map(|s| s.to_string())
                .collect();
            current_words = overlap_words.len();
            current_chunk = overlap_words.join(" ");
        }

        i += 1;
    }

    if !current_chunk.trim().is_empty() {
        chunks.push(current_chunk.trim().to_string());
    }

    if chunks.is_empty() {
        vec![text.to_string()]
    } else {
        chunks
    }
}

/// Clean and shorten a passage for readable output
fn clean_passage(s: &str, max_chars: usize) -> String {
    // Collapse multiple spaces
    let cleaned: String = s
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");

    if cleaned.len() <= max_chars {
        cleaned
    } else {
        // Try to cut at a word boundary
        let cut = &cleaned[..max_chars];
        if let Some(last_space) = cut.rfind(' ') {
            format!("{}...", &cleaned[..last_space])
        } else {
            format!("{}...", cut)
        }
    }
}

fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    let i = a.intersection(b).count();
    let u = a.union(b).count();
    if u == 0 { 0.0 } else { i as f64 / u as f64 }
}

/// Bonus score for calendar-domain keywords appearing in both question and chunk
fn calendar_keyword_bonus(q: &str, chunk: &str) -> f64 {
    let terms = [
        // Events
        "graduation", "convocation", "research festival", "open day",
        "hdc", "higher degrees", "senate", "council",
        // Terms
        "term 1", "term 2", "term 3", "term 4",
        "start of term", "end of term",
        // Months
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        // Years
        "2024", "2025", "2026",
        // Holidays
        "good friday", "christmas", "heritage day", "workers day",
        "youth day", "womens day", "freedom day", "reconciliation",
        "human rights", "africa day", "mandela day",
        // Academic
        "academic staff", "administrative staff", "wced", "first year",
    ];

    let bonus: f64 = terms
        .iter()
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