use crate::data::{generate_qa_pairs, train_val_split, DocumentLoader, QAExample};
use crate::model::{QATransformer, QATransformerConfig};
use crate::tokenizer::SimpleTokenizer;
use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub epochs: usize,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub data_dir: String,
    pub model_output: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub train_loss: f64,
    pub val_loss: f64,
    pub train_accuracy: f64,
    pub val_accuracy: f64,
    pub perplexity: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelCheckpoint {
    pub config: TrainingConfig,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub d_model: usize,
    pub num_layers: usize,
    pub metrics_history: Vec<EpochMetrics>,
    pub tokenizer_path: String,
    pub documents: Vec<crate::data::Document>,
    pub qa_examples: Vec<QAExample>,
}

pub fn train(config: TrainingConfig) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n[1/5] Loading .docx documents from '{}'...", config.data_dir);
    let loader = DocumentLoader::new(&config.data_dir);
    let documents = loader.load_all()?;
    println!("  Loaded {} documents", documents.len());

    println!("\n[2/5] Generating Q&A training pairs...");
    let mut qa_pairs = generate_qa_pairs(&documents);
    println!("  Generated {} Q&A examples", qa_pairs.len());

    println!("\n[3/5] Building tokenizer...");
    let mut tokenizer = SimpleTokenizer::new(256);
    let all_texts: Vec<&str> = qa_pairs
        .iter()
        .flat_map(|qa| vec![qa.context.as_str(), qa.question.as_str(), qa.answer.as_str()])
        .collect();
    tokenizer.build_vocab(&all_texts);

    for qa in &mut qa_pairs {
        qa.input_ids = tokenizer.encode_qa(&qa.context, &qa.question, Some(256));
        qa.label_ids = tokenizer.encode(&qa.answer, Some(256));
    }

    let (train_set, val_set) = train_val_split(qa_pairs.clone(), 0.15);
    println!("  Train: {} | Val: {}", train_set.len(), val_set.len());

    let tokenizer_path = "tokenizer.json";
    tokenizer.save(tokenizer_path)?;

    println!("\n[4/5] Initializing transformer model...");
    let model_config = QATransformerConfig::default_config(tokenizer.vocab_size);
    println!(
        "  d_model={}, layers={}, heads={}, vocab={}, params={}",
        model_config.d_model,
        model_config.num_layers,
        model_config.num_heads,
        model_config.vocab_size,
        estimate_params(&model_config)
    );

    println!(
        "\n[5/5] Training ({} epochs, lr={}, batch={})...",
        config.epochs, config.learning_rate, config.batch_size
    );

    let metrics_history = run_training_loop(&train_set, &val_set, &model_config, &config)?;

    let checkpoint = ModelCheckpoint {
        config: config.clone(),
        vocab_size: tokenizer.vocab_size,
        max_seq_len: model_config.max_seq_len,
        d_model: model_config.d_model,
        num_layers: model_config.num_layers,
        metrics_history,
        tokenizer_path: tokenizer_path.to_string(),
        documents,
        qa_examples: qa_pairs,
    };

    fs::write(&config.model_output, serde_json::to_string_pretty(&checkpoint)?)?;
    println!("\nCheckpoint saved: '{}'", config.model_output);
    Ok(())
}

fn run_training_loop(
    train_set: &[QAExample],
    val_set: &[QAExample],
    model_config: &QATransformerConfig,
    training_config: &TrainingConfig,
) -> Result<Vec<EpochMetrics>, Box<dyn std::error::Error>> {
    // NOTE:
    // Training on the WGPU backend can trigger very deep Send/Sync trait evaluation
    // (wgpu-core validation types) on some toolchains.
    // To keep the project compiling reliably, we train on the CPU (NdArray) backend.
    // You can still keep the required Burn features (wgpu/train/autodiff) enabled in Cargo.toml.
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::{Autodiff, NdArray};
    use burn::optim::{AdamConfig, GradientsParams, Optimizer};

    type B = Autodiff<NdArray>;
    let device = NdArrayDevice::default();

    let mut model: QATransformer<B> = QATransformer::new(model_config, &device);
    let mut optim = AdamConfig::new().with_epsilon(1e-8).init();

    let batch_size = training_config.batch_size.max(1);
    let mut history = Vec::new();

    for epoch in 1..=training_config.epochs {
        let mut total_loss = 0.0f64;
        let mut total_correct = 0usize;
        let mut total_tokens = 0usize;
        let mut num_batches = 0;

        for batch in train_set.chunks(batch_size) {
            let (inputs, labels) = prepare_batch::<B>(batch, &device);
            let logits = model.forward(inputs, true);
            let (loss, correct, tokens) = cross_entropy::<B>(logits, labels);

            total_loss += scalar_to_f64(loss.clone().into_scalar());
            total_correct += correct;
            total_tokens += tokens;
            num_batches += 1;

            let grads = loss.backward();
            let grad_params = GradientsParams::from_grads(grads, &model);
            model = optim.step(training_config.learning_rate, model, grad_params);
        }

        let (val_loss, val_acc) = eval_loop(&model, val_set, &device);
        let train_loss = total_loss / num_batches.max(1) as f64;
        let train_acc = total_correct as f64 / total_tokens.max(1) as f64;
        let perplexity = val_loss.exp();

        println!(
            "  Epoch {:3}/{} | TLoss={:.4} VLoss={:.4} | TAcc={:.1}% VAcc={:.1}% | PPL={:.2}",
            epoch, training_config.epochs,
            train_loss, val_loss,
            train_acc * 100.0, val_acc * 100.0,
            perplexity,
        );

        history.push(EpochMetrics {
            epoch,
            train_loss,
            val_loss,
            train_accuracy: train_acc,
            val_accuracy: val_acc,
            perplexity,
        });
    }

    Ok(history)
}

fn eval_loop<B: burn::tensor::backend::AutodiffBackend>(
    model: &QATransformer<B>,
    dataset: &[QAExample],
    device: &B::Device,
) -> (f64, f64) {
    if dataset.is_empty() {
        return (0.0, 0.0);
    }

    let mut total_loss = 0.0f64;
    let mut total_correct = 0usize;
    let mut total_tokens = 0usize;
    let mut num_batches = 0;

    for batch in dataset.chunks(4) {
        let (inputs, labels) = prepare_batch::<B>(batch, device);
        let logits = model.forward(inputs, false);
        let (loss, correct, tokens) = cross_entropy::<B>(logits, labels);
        total_loss += scalar_to_f64(loss.into_scalar());
        total_correct += correct;
        total_tokens += tokens;
        num_batches += 1;
    }

    let avg_loss = total_loss / num_batches.max(1) as f64;
    let accuracy = total_correct as f64 / total_tokens.max(1) as f64;
    (avg_loss, accuracy)
}

fn cross_entropy<B: burn::tensor::backend::AutodiffBackend>(
    logits: burn::tensor::Tensor<B, 3>,
    labels: burn::tensor::Tensor<B, 2, burn::tensor::Int>,
) -> (burn::tensor::Tensor<B, 1>, usize, usize) {
    use burn::tensor::activation::log_softmax;

    let [batch, seq_len, vocab] = logits.dims();
    let total = batch * seq_len;

    let logits_flat = logits.clone().reshape([total, vocab]);
    let labels_flat = labels.clone().reshape([total]);

    // Proper negative log-likelihood:
    // pick log-prob of the true label for each position then average.
    let log_probs = log_softmax(logits_flat.clone(), 1); // [total, vocab]
    let indices = labels_flat.clone().unsqueeze_dim::<2>(1); // [total, 1]
    let picked = log_probs.gather(1, indices).squeeze::<1>(); // [total]
    let loss = picked.neg().mean();

    // argmax returns [total, 1] — use squeeze::<1>() with no argument
    let preds = logits_flat.argmax(1).squeeze::<1>();
    let correct = scalar_to_f64(
        preds.equal(labels_flat).int().sum().into_scalar()
    ) as usize;

    (loss, correct, total)
}

fn prepare_batch<B: burn::tensor::backend::Backend>(
    batch: &[QAExample],
    device: &B::Device,
) -> (
    burn::tensor::Tensor<B, 2, burn::tensor::Int>,
    burn::tensor::Tensor<B, 2, burn::tensor::Int>,
) {
    use burn::tensor::{Int, Tensor};

    let max_in = batch.iter().map(|e| e.input_ids.len()).max().unwrap_or(1);
    let max_la = batch.iter().map(|e| e.label_ids.len()).max().unwrap_or(1);
    let b = batch.len();

    let mut input_data = Vec::with_capacity(b * max_in);
    let mut label_data = Vec::with_capacity(b * max_la);

    for ex in batch {
        let mut inp: Vec<i32> = ex.input_ids.iter().map(|&x| x as i32).collect();
        inp.resize(max_in, 0);
        input_data.extend_from_slice(&inp[..max_in]);

        let mut lab: Vec<i32> = ex.label_ids.iter().map(|&x| x as i32).collect();
        lab.resize(max_la, 0);
        label_data.extend_from_slice(&lab[..max_la]);
    }

    let inputs = Tensor::<B, 1, Int>::from_ints(input_data.as_slice(), device)
        .reshape([b, max_in]);
    let labels = Tensor::<B, 1, Int>::from_ints(label_data.as_slice(), device)
        .reshape([b, max_la]);

    (inputs, labels)
}

/// Safely convert any Burn scalar to f64 via string
fn scalar_to_f64<T: std::fmt::Display>(val: T) -> f64 {
    val.to_string().parse::<f64>().unwrap_or(0.0)
}

fn estimate_params(cfg: &QATransformerConfig) -> String {
    let tok_emb = cfg.vocab_size * cfg.d_model;
    let pos_emb = cfg.max_seq_len * cfg.d_model;
    let per_layer = 4 * cfg.d_model * cfg.d_model + 2 * cfg.d_model * cfg.d_ff + 4 * cfg.d_model;
    let total = tok_emb + pos_emb + cfg.num_layers * per_layer + cfg.d_model * cfg.vocab_size;
    if total >= 1_000_000 {
        format!("~{:.2}M", total as f64 / 1_000_000.0)
    } else {
        format!("~{}K", total / 1000)
    }
}