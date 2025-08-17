use nalgebra::{DMatrix, DVector};
use nalgebra::linalg::SVD;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TfIdfLsa {
    vocabulary: HashMap<String, usize>,
    idf_weights: Vec<f32>,
    lsa_components: Option<DMatrix<f32>>,
    embedding_dim: usize,
    documents_count: usize,
}

impl TfIdfLsa {
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            vocabulary: HashMap::new(),
            idf_weights: Vec::new(),
            lsa_components: None,
            embedding_dim,
            documents_count: 0,
        }
    }

    // Build TF-IDF matrix from documents
    pub fn fit(&mut self, documents: &[Vec<String>], vocabulary: HashMap<String, usize>) {
        self.vocabulary = vocabulary;
        self.documents_count = documents.len();
        
        let vocab_size = self.vocabulary.len();
        
        // Calculate document frequencies
        let mut doc_freq = vec![0usize; vocab_size];
        for doc_tokens in documents {
            let mut seen = vec![false; vocab_size];
            for token in doc_tokens {
                if let Some(&idx) = self.vocabulary.get(token) {
                    if !seen[idx] {
                        doc_freq[idx] += 1;
                        seen[idx] = true;
                    }
                }
            }
        }
        
        // Calculate IDF weights
        self.idf_weights = doc_freq
            .iter()
            .map(|&df| {
                if df > 0 {
                    ((self.documents_count as f32 + 1.0) / (df as f32 + 1.0)).ln()
                } else {
                    0.0
                }
            })
            .collect();
        
        // Build TF-IDF matrix
        let mut tfidf_matrix = DMatrix::zeros(vocab_size, self.documents_count);
        
        for (doc_idx, doc_tokens) in documents.iter().enumerate() {
            // Calculate term frequencies
            let mut tf_counts = vec![0f32; vocab_size];
            for token in doc_tokens {
                if let Some(&idx) = self.vocabulary.get(token) {
                    tf_counts[idx] += 1.0;
                }
            }
            
            // Normalize TF and apply IDF
            let total_terms = doc_tokens.len() as f32;
            for (term_idx, &count) in tf_counts.iter().enumerate() {
                if count > 0.0 {
                    let tf = count / total_terms;
                    let tfidf = tf * self.idf_weights[term_idx];
                    tfidf_matrix[(term_idx, doc_idx)] = tfidf;
                }
            }
        }
        
        // Perform LSA using SVD
        if self.documents_count >= 2 && vocab_size >= self.embedding_dim {
            self.perform_lsa(tfidf_matrix);
        }
    }
    
    // Perform Latent Semantic Analysis using SVD
    fn perform_lsa(&mut self, tfidf_matrix: DMatrix<f32>) {
        let (nrows, ncols) = tfidf_matrix.shape();
        let target_dim = self.embedding_dim.min(nrows).min(ncols);
        
        // Perform Singular Value Decomposition (SVD)
        // TF-IDF matrix = U * Σ * V^T
        // Where U contains left singular vectors (document-concept relationships)
        // Σ contains singular values (importance of each concept)
        // V^T contains right singular vectors (term-concept relationships)
        let svd = SVD::new(tfidf_matrix.clone(), true, true);
        
        // Extract U matrix (left singular vectors)
        if let Some(u_matrix) = svd.u {
            // Select top k components from U matrix
            // These represent the most important latent semantic dimensions
            let mut components = DMatrix::zeros(target_dim, nrows);
            
            // Copy the first target_dim columns of U^T
            // We transpose because we want each row to be a component
            for i in 0..target_dim {
                for j in 0..nrows {
                    components[(i, j)] = u_matrix[(j, i)];
                }
            }
            
            // Optional: Weight components by singular values for better representation
            // This gives more importance to stronger latent dimensions
            let singular_values = svd.singular_values;
            for i in 0..target_dim.min(singular_values.len()) {
                let weight = singular_values[i].sqrt();
                for j in 0..nrows {
                    components[(i, j)] *= weight;
                }
            }
            
            self.lsa_components = Some(components);
        } else {
            // Fallback to identity-like transformation if SVD fails
            let mut components = DMatrix::zeros(target_dim, nrows);
            for i in 0..target_dim.min(nrows) {
                components[(i, i)] = 1.0;
            }
            self.lsa_components = Some(components);
        }
    }
    
    // Transform a document to embedding vector
    pub fn transform(&self, tokens: &[String]) -> Vec<f32> {
        let vocab_size = self.vocabulary.len();
        
        // Return zero vector if vocabulary is empty
        if vocab_size == 0 {
            return vec![0.0; self.embedding_dim];
        }
        
        // Calculate TF-IDF vector for the document
        let mut tfidf_vec = vec![0f32; vocab_size];
        let mut tf_counts = vec![0f32; vocab_size];
        
        // Count term frequencies
        for token in tokens {
            if let Some(&idx) = self.vocabulary.get(token) {
                tf_counts[idx] += 1.0;
            }
        }
        
        // Normalize and apply IDF
        let total_terms = tokens.len() as f32;
        if total_terms > 0.0 {
            for (idx, &count) in tf_counts.iter().enumerate() {
                if count > 0.0 && idx < self.idf_weights.len() {
                    let tf = count / total_terms;
                    tfidf_vec[idx] = tf * self.idf_weights[idx];
                }
            }
        }
        
        // Apply LSA transformation if available
        if let Some(ref components) = self.lsa_components {
            let tfidf_vector = DVector::from_vec(tfidf_vec);
            let embedded = components * tfidf_vector;
            embedded.iter().cloned().collect()
        } else {
            // Return truncated TF-IDF vector if LSA not available
            tfidf_vec.truncate(self.embedding_dim);
            tfidf_vec.resize(self.embedding_dim, 0.0);
            tfidf_vec
        }
    }
    
    // Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocabulary.len()
    }
    
    // Get embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
    
    // Export model to JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
    
    // Import model from JSON
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::JapaneseTokenizer;
    
    #[test]
    fn test_tfidf_lsa_basic() {
        let tokenizer = JapaneseTokenizer::new();
        let documents = vec![
            "今日は天気がいいですね",
            "明日は雨が降りそうです",
            "今日は映画を見ました",
        ];
        
        // Tokenize documents
        let tokenized_docs: Vec<Vec<String>> = documents
            .iter()
            .map(|doc| tokenizer.tokenize(doc))
            .collect();
        
        // Build vocabulary
        let vocab = tokenizer.build_vocabulary(&documents.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        
        // Create and fit TF-IDF LSA model
        let mut model = TfIdfLsa::new(64);
        model.fit(&tokenized_docs, vocab);
        
        // Transform a document
        let test_doc = "今日は晴れです";
        let test_tokens = tokenizer.tokenize(test_doc);
        let embedding = model.transform(&test_tokens);
        
        // Check embedding dimension
        assert_eq!(embedding.len(), 64);
        
        // Check that embedding is not all zeros
        let sum: f32 = embedding.iter().map(|x| x.abs()).sum();
        assert!(sum > 0.0);
    }
    
    #[test]
    fn test_model_serialization() {
        let mut model = TfIdfLsa::new(32);
        let vocab = HashMap::from([
            ("今日".to_string(), 0),
            ("明日".to_string(), 1),
            ("天気".to_string(), 2),
        ]);
        
        let documents = vec![
            vec!["今日".to_string(), "天気".to_string()],
            vec!["明日".to_string(), "天気".to_string()],
        ];
        
        model.fit(&documents, vocab);
        
        // Serialize to JSON
        let json = model.to_json().unwrap();
        
        // Deserialize from JSON
        let restored = TfIdfLsa::from_json(&json).unwrap();
        
        // Check that restored model has same properties
        assert_eq!(model.vocab_size(), restored.vocab_size());
        assert_eq!(model.embedding_dim(), restored.embedding_dim());
    }
}