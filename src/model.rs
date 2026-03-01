use burn::{
    config::Config,
    module::Module,
    nn::{
        attention::{MultiHeadAttention, MultiHeadAttentionConfig, MhaInput},
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear,
        LinearConfig,
    },
    tensor::{backend::Backend, activation, Tensor, Int},
};

#[derive(Config, Debug)]
pub struct QATransformerConfig {
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub d_ff: usize,
    #[config(default = "0.1")]
    pub dropout: f64,
}

impl QATransformerConfig {
    pub fn default_config(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            max_seq_len: 256,
            d_model: 128,
            num_heads: 4,
            num_layers: 6,
            d_ff: 512,
            dropout: 0.1,
        }
    }

    pub fn large_config(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            max_seq_len: 256,
            d_model: 256,
            num_heads: 8,
            num_layers: 8,
            d_ff: 1024,
            dropout: 0.1,
        }
    }
}

#[derive(Module, Debug)]
pub struct TransformerEncoderLayer<B: Backend> {
    self_attention: MultiHeadAttention<B>,
    norm1: LayerNorm<B>,
    ff1: Linear<B>,
    ff2: Linear<B>,
    norm2: LayerNorm<B>,
    dropout: Dropout,
}

impl<B: Backend> TransformerEncoderLayer<B> {
    pub fn new(config: &QATransformerConfig, device: &B::Device) -> Self {
        Self {
            self_attention: MultiHeadAttentionConfig::new(config.d_model, config.num_heads)
                .with_dropout(config.dropout) // f64 directly
                .init(device),
            norm1: LayerNormConfig::new(config.d_model).init(device),
            norm2: LayerNormConfig::new(config.d_model).init(device),
            ff1: LinearConfig::new(config.d_model, config.d_ff).init(device),
            ff2: LinearConfig::new(config.d_ff, config.d_model).init(device),
            dropout: DropoutConfig::new(config.dropout).init(), // f64 directly
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, training: bool) -> Tensor<B, 3> {
        let normed = self.norm1.forward(x.clone());
        let attn = self.self_attention.forward(MhaInput::self_attn(normed)).context;
        let attn = if training { self.dropout.forward(attn) } else { attn };
        let x = x + attn;

        let normed = self.norm2.forward(x.clone());
        let ff = activation::relu(self.ff1.forward(normed));
        let ff = self.ff2.forward(ff);
        let ff = if training { self.dropout.forward(ff) } else { ff };

        x + ff
    }
}

#[derive(Module, Debug)]
pub struct PositionalEncoding<B: Backend> {
    embedding: Embedding<B>,
    d_model: usize,
}

impl<B: Backend> PositionalEncoding<B> {
    pub fn new(max_seq_len: usize, d_model: usize, device: &B::Device) -> Self {
        Self {
            embedding: EmbeddingConfig::new(max_seq_len, d_model).init(device),
            d_model,
        }
    }

    pub fn forward(&self, seq_len: usize, batch_size: usize, device: &B::Device) -> Tensor<B, 3> {
        let positions: Vec<i32> = (0..seq_len as i32).collect();
        let pos = Tensor::<B, 1, Int>::from_ints(positions.as_slice(), device).unsqueeze::<2>();
        self.embedding.forward(pos).expand([batch_size, seq_len, self.d_model])
    }
}

#[derive(Module, Debug)]
pub struct QATransformer<B: Backend> {
    token_embedding: Embedding<B>,
    positional_encoding: PositionalEncoding<B>,
    encoder_layers: Vec<TransformerEncoderLayer<B>>,
    output_projection: Linear<B>,
    final_norm: LayerNorm<B>,
    dropout: Dropout,
    d_model: usize,
}

impl<B: Backend> QATransformer<B> {
    pub fn new(config: &QATransformerConfig, device: &B::Device) -> Self {
        Self {
            token_embedding: EmbeddingConfig::new(config.vocab_size, config.d_model).init(device),
            positional_encoding: PositionalEncoding::new(
                config.max_seq_len, config.d_model, device,
            ),
            encoder_layers: (0..config.num_layers)
                .map(|_| TransformerEncoderLayer::new(config, device))
                .collect(),
            output_projection: LinearConfig::new(config.d_model, config.vocab_size)
                .with_bias(false)
                .init(device),
            final_norm: LayerNormConfig::new(config.d_model).init(device),
            dropout: DropoutConfig::new(config.dropout).init(), // f64 directly
            d_model: config.d_model,
        }
    }

    pub fn forward(&self, input_ids: Tensor<B, 2, Int>, training: bool) -> Tensor<B, 3> {
        let [batch_size, seq_len] = input_ids.dims();
        let device = &input_ids.device();

        let scale = (self.d_model as f64).sqrt() as f32;
        let tok_emb = self.token_embedding.forward(input_ids) * scale;
        let pos_emb = self.positional_encoding.forward(seq_len, batch_size, device);

        let mut h: Tensor<B, 3> = tok_emb + pos_emb;
        if training {
            h = self.dropout.forward(h);
        }

        for layer in &self.encoder_layers {
            h = layer.forward(h, training);
        }

        self.output_projection.forward(self.final_norm.forward(h))
    }
}