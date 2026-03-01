use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub filename: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QAExample {
    pub context: String,
    pub question: String,
    pub answer: String,
    pub input_ids: Vec<u32>,
    pub label_ids: Vec<u32>,
}

pub struct DocumentLoader {
    data_dir: PathBuf,
}

impl DocumentLoader {
    pub fn new(data_dir: &str) -> Self {
        Self { data_dir: PathBuf::from(data_dir) }
    }

    pub fn load_all(&self) -> Result<Vec<Document>, Box<dyn std::error::Error>> {
        let mut documents = Vec::new();

        if !self.data_dir.exists() {
            return Err(format!(
                "Data directory '{}' not found.",
                self.data_dir.display()
            ).into());
        }

        for entry in fs::read_dir(&self.data_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|e| e.to_str()) == Some("docx") {
                let filename = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string();

                match self.extract_text(&path) {
                    Ok(content) if !content.trim().is_empty() => {
                        println!("  Loaded: {} ({} chars)", filename, content.len());
                        documents.push(Document { filename, content });
                    }
                    Ok(_) => eprintln!("  Warning: {} produced empty content", filename),
                    Err(e) => eprintln!("  Warning: Could not parse {}: {}", filename, e),
                }
            }
        }

        if documents.is_empty() {
            return Err("No readable .docx files found in data directory".into());
        }

        Ok(documents)
    }

    fn extract_text(&self, path: &Path) -> Result<String, Box<dyn std::error::Error>> {
        use docx_rs::*;
        let bytes = fs::read(path)?;
        let docx = read_docx(&bytes)?;
        // Serialize entire docx to JSON then recursively collect all "text" fields.
        // This avoids direct struct field access that breaks across docx_rs versions.
        let json_val = serde_json::to_value(&docx)?;
        Ok(extract_text_from_json(&json_val))
    }
}

/// Walk a serde_json Value tree and collect every string stored under key "text".
fn extract_text_from_json(val: &serde_json::Value) -> String {
    match val {
        serde_json::Value::Object(map) => {
            if let Some(serde_json::Value::String(s)) = map.get("text") {
                let t = s.trim();
                if !t.is_empty() {
                    return t.to_string();
                }
            }
            map.values()
                .map(extract_text_from_json)
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join(" ")
        }
        serde_json::Value::Array(arr) => {
            arr.iter()
                .map(extract_text_from_json)
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join(" ")
        }
        _ => String::new(),
    }
}

pub fn generate_qa_pairs(_documents: &[Document]) -> Vec<QAExample> {
    let ctx = String::from("CPUT Academic Calendar 2024-2026");

    let raw: &[(&str, &str)] = &[
        ("What is the Month and date will the 2026 End of year Graduation Ceremony be held?",
         "The 2026 End of Year Graduation Ceremony: The Convocation Executive Meeting is on Saturday 1 August 2026."),
        ("How many times did the HDC hold their meetings in 2024?",
         "The Higher Degrees Committee (HDC) held 8 meetings in 2024: February (19th), March (5th), May (2nd), July (22nd), August (7th), October (17th), and November (12th)."),
        ("When does Term 1 start in 2026?", "Term 1 of 2026 starts on Monday 26 January 2026."),
        ("When does Term 2 start in 2026?", "Term 2 of 2026 starts on Monday 23 March 2026."),
        ("When does Term 3 start in 2026?", "Term 3 of 2026 starts on Monday 13 July 2026."),
        ("When does Term 4 start in 2026?", "Term 4 of 2026 starts on Monday 14 September 2026."),
        ("When does Term 1 end in 2026?", "Term 1 ends on Friday 13 March 2026."),
        ("When does Term 2 end in 2026?", "Term 2 ends on Friday 19 June 2026."),
        ("When does Term 3 end in 2026?", "Term 3 ends on Friday 4 September 2026."),
        ("When does Term 4 end in 2026?", "Term 4 ends on Friday 11 December 2026."),
        ("What is the date of the 2026 End of Year Graduation Ceremony?",
         "The 2026 End of Year Graduation Convocation Executive Meeting is on Saturday 1 August 2026."),
        ("When is the 2026 graduation ceremony?",
         "The 2026 Graduation Convocation Executive Meeting is on Saturday 1 August 2026."),
        ("When does the academic year start for Administrative Staff in 2026?",
         "Administrative Staff start on Wednesday 7 January 2026."),
        ("When does the academic year start for Academic Staff in 2026?",
         "Academic Staff start on Monday 12 January 2026."),
        ("When is New Year's Day in 2026?", "New Year's Day 2026 is on Thursday 1 January 2026."),
        ("When is Good Friday in 2026?", "Good Friday in 2026 is on Friday 3 April 2026."),
        ("When is Workers Day in 2026?", "Workers Day in 2026 is on Friday 1 May 2026."),
        ("When is Youth Day in 2026?", "Youth Day in 2026 is on Tuesday 16 June 2026."),
        ("When is Mandela Day in 2026?", "Mandela Day in 2026 is on Saturday 18 July 2026."),
        ("When is Women's Day in 2026?", "Women's Day in 2026 is observed on Monday 10 August 2026."),
        ("When is Heritage Day in 2026?", "Heritage Day in 2026 is on Thursday 24 September 2026."),
        ("When is Christmas Day in 2026?", "Christmas Day in 2026 is on Friday 25 December 2026."),
        ("When is Freedom Day in 2026?", "Freedom Day in 2026 is on Monday 27 April 2026."),
        ("When is Africa Day in 2026?", "Africa Day in 2026 is on Monday 25 May 2026."),
        ("When is Human Rights Day in 2026?", "Human Rights Day in 2026 is on Saturday 21 March 2026."),
        ("When is the Day of Reconciliation in 2026?", "Day of Reconciliation in 2026 is on Wednesday 16 December 2026."),
        ("When does WCED Schools open in January 2026?", "WCED Schools open on Wednesday 14 January 2026."),
        ("When is the Research Festival in 2026?",
         "The Research Festival 2026: Day 1 (Showcase and Awards) on 18 August, Day 2 (Postgraduate Conference) on 19 August, Day 3 (Ethics Day) on 20 August 2026."),
        ("When does Term 1 start in 2025?", "Term 1 of 2025 starts on Monday 27 January 2025."),
        ("When does Term 1 start in 2024?", "Term 1 of 2024 starts on Monday 29 January 2024."),
        ("When does End of Year for Academic Staff occur in 2026?",
         "End of Year for Academic Staff in 2026 is on Friday 11 December 2026."),
        ("When does End of Year for Administrative Staff occur in 2026?",
         "End of Year for Administrative Staff in 2026 is on Friday 18 December 2026."),
        ("When is the Annual Open Day in 2026?", "The Annual Open Day in 2026 is on Saturday 9 May 2026 at 09:00."),
        ("When is the Day of Goodwill in 2026?", "The Day of Goodwill in 2026 is on Saturday 26 December 2026."),
        ("When does WCED Schools close in June 2026?", "WCED Schools close on Friday 26 June 2026."),
    ];

    raw.iter()
        .map(|(q, a)| QAExample {
            context: ctx.clone(),
            question: q.to_string(),
            answer: a.to_string(),
            input_ids: Vec::new(),
            label_ids: Vec::new(),
        })
        .collect()
}

pub fn train_val_split(
    mut examples: Vec<QAExample>,
    val_ratio: f32,
) -> (Vec<QAExample>, Vec<QAExample>) {
    let total = examples.len();
    let val_count = ((total as f32) * val_ratio).ceil() as usize;
    let train_count = total.saturating_sub(val_count);
    let val = examples.split_off(train_count);
    (examples, val)
}