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
        Self {
            data_dir: PathBuf::from(data_dir),
        }
    }

    pub fn load_all(&self) -> Result<Vec<Document>, Box<dyn std::error::Error>> {
        let mut documents = Vec::new();

        if !self.data_dir.exists() {
            return Err(format!(
                "Data directory '{}' not found.",
                self.data_dir.display()
            )
            .into());
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
        let mut parts: Vec<String> = Vec::new();

        for child in &docx.document.children {
            match child {
                DocumentChild::Paragraph(para) => {
                    let mut para_parts = Vec::new();

                    for pc in &para.children {
                        if let ParagraphChild::Run(run) = pc {
                            for rc in &run.children {
                                if let RunChild::Text(t) = rc {
                                    let s = t.text.trim();
                                    if !s.is_empty() {
                                        para_parts.push(s.to_string());
                                    }
                                }
                            }
                        }
                    }

                    if !para_parts.is_empty() {
                        parts.push(para_parts.join(" "));
                    }
                }

      DocumentChild::Table(table) => {
    for row in &table.rows {
        if let TableChild::TableRow(tr) = row {
            for cell in &tr.cells {
                if let TableRowChild::TableCell(tc) = cell {
                    for cc in &tc.children {
                        if let TableCellContent::Paragraph(para) = cc {
                            let mut cell_parts = Vec::new();

                            for pc in &para.children {
                                if let ParagraphChild::Run(run) = pc {
                                    for rc in &run.children {
                                        if let RunChild::Text(t) = rc {
                                            let s = t.text.trim();
                                            if !s.is_empty() {
                                                cell_parts.push(s.to_string());
                                            }
                                        }
                                    }
                                }
                            }

                            if !cell_parts.is_empty() {
                                parts.push(cell_parts.join(" "));
                            }
                        }
                    }
                }
            }
        }
    }
}

                _ => {}
            }
        }

        Ok(parts.join(" "))
    }
}

pub fn generate_qa_pairs(documents: &[Document]) -> Vec<QAExample> {
    let mut examples = Vec::new();

    let combined = documents
        .iter()
        .map(|d| format!("[{}]\n{}", d.filename, d.content))
        .collect::<Vec<_>>()
        .join("\n\n");

    let global_ctx = combined[..combined.len().min(2000)].to_string();

    let months = [
        "january","february","march","april","may","june",
        "july","august","september","october","november","december",
    ];

    fn clean_line(s: &str) -> String {
        s.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    for doc in documents {
        let ctx = doc.content[..doc.content.len().min(1200)].to_string();

        for raw_line in doc.content.lines() {
            let line = clean_line(raw_line);
            if line.len() < 20 {
                continue;
            }

            let low = line.to_lowercase();
            let has_year = low.contains("2024") || low.contains("2025") || low.contains("2026");
            let has_month = months.iter().any(|m| low.contains(m));

            if !(has_year && has_month) {
                continue;
            }

            let q_templates = [
                format!("What is the date for: {}?", line),
                format!("When is {}?", line),
                format!("Which day and date is mentioned for: {}?", line),
            ];

            for q in q_templates {
                examples.push(QAExample {
                    context: ctx.clone(),
                    question: q,
                    answer: line.clone(),
                    input_ids: Vec::new(),
                    label_ids: Vec::new(),
                });
            }

            if examples.len() > 600 {
                break;
            }
        }
    }

    if examples.is_empty() {
        examples.push(QAExample {
            context: global_ctx,
            question: "What information is contained in the calendar documents?".to_string(),
            answer: "The documents contain calendar events, term dates, holidays, and meeting schedules for 2024–2026.".to_string(),
            input_ids: Vec::new(),
            label_ids: Vec::new(),
        });
    }

    examples
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