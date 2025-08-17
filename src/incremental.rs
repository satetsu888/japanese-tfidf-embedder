use crate::tokenizer::JapaneseTokenizer;
use crate::tfidf_lsa::TfIdfLsa;
use crate::utils::{cosine_similarity, l2_normalize};
use serde::{Deserialize, Serialize};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

// Define error type for non-wasm targets
#[cfg(not(target_arch = "wasm32"))]
type JsValue = String;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
#[derive(Clone, Serialize, Deserialize)]
pub struct IncrementalEmbedder {
    tokenizer: JapaneseTokenizer,
    model: TfIdfLsa,
    documents: Vec<String>,
    tokenized_documents: Vec<Vec<String>>,
    update_threshold: f32,
    changes_since_update: usize,
    is_retraining: bool,
    retrain_progress: f32,
    
    // For background retraining
    pending_model: Option<TfIdfLsa>,
    retrain_step: RetrainStep,
}

#[derive(Clone, Serialize, Deserialize)]
enum RetrainStep {
    Idle,
    BuildingVocabulary,
    ComputingTfIdf,
    PerformingSvd,
    Complete,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl IncrementalEmbedder {
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(constructor))]
    pub fn new(update_threshold: f32) -> Self {
        Self {
            tokenizer: JapaneseTokenizer::new(),
            model: TfIdfLsa::new(64),
            documents: Vec::new(),
            tokenized_documents: Vec::new(),
            update_threshold,
            changes_since_update: 0,
            is_retraining: false,
            retrain_progress: 0.0,
            pending_model: None,
            retrain_step: RetrainStep::Idle,
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn new_with_ngrams(update_threshold: f32, min_ngram: usize, max_ngram: usize) -> Self {
        Self {
            tokenizer: JapaneseTokenizer::new_with_ngrams(min_ngram, max_ngram),
            model: TfIdfLsa::new(64),
            documents: Vec::new(),
            tokenized_documents: Vec::new(),
            update_threshold,
            changes_since_update: 0,
            is_retraining: false,
            retrain_progress: 0.0,
            pending_model: None,
            retrain_step: RetrainStep::Idle,
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn add_document(&mut self, text: String, embedding_dim: usize) -> Result<(), JsValue> {
        // Add document to collection
        self.documents.push(text.clone());
        let tokens = self.tokenizer.tokenize(&text);
        self.tokenized_documents.push(tokens);
        
        self.changes_since_update += 1;
        
        // Check if we need to retrain
        let change_ratio = self.changes_since_update as f32 / self.documents.len().max(1) as f32;
        if change_ratio >= self.update_threshold && !self.is_retraining {
            self.start_background_retrain(embedding_dim)?;
        }
        
        Ok(())
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn transform(&self, text: &str) -> Result<Vec<f32>, JsValue> {
        let tokens = self.tokenizer.tokenize(text);
        let mut embedding = self.model.transform(&tokens);
        l2_normalize(&mut embedding);
        Ok(embedding)
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn start_background_retrain(&mut self, embedding_dim: usize) -> Result<(), JsValue> {
        if self.is_retraining {
            #[cfg(target_arch = "wasm32")]
            return Err(JsValue::from_str("Retraining already in progress"));
            #[cfg(not(target_arch = "wasm32"))]
            return Err("Retraining already in progress".to_string());
        }
        
        self.is_retraining = true;
        self.retrain_progress = 0.0;
        self.retrain_step = RetrainStep::BuildingVocabulary;
        self.pending_model = Some(TfIdfLsa::new(embedding_dim));
        
        Ok(())
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn step_retrain(&mut self) -> Result<bool, JsValue> {
        if !self.is_retraining {
            return Ok(true);
        }
        
        match self.retrain_step {
            RetrainStep::Idle => Ok(true),
            
            RetrainStep::BuildingVocabulary => {
                // Build vocabulary (simulated as single step for simplicity)
                let vocab = self.tokenizer.build_vocabulary(&self.documents);
                
                if let Some(ref mut pending_model) = self.pending_model {
                    // Store vocabulary for next step
                    pending_model.fit(&self.tokenized_documents, vocab);
                }
                
                self.retrain_progress = 0.33;
                self.retrain_step = RetrainStep::ComputingTfIdf;
                Ok(false)
            }
            
            RetrainStep::ComputingTfIdf => {
                // TF-IDF computation is done in fit()
                self.retrain_progress = 0.66;
                self.retrain_step = RetrainStep::PerformingSvd;
                Ok(false)
            }
            
            RetrainStep::PerformingSvd => {
                // SVD is done in fit(), so we just finalize
                self.retrain_progress = 1.0;
                self.retrain_step = RetrainStep::Complete;
                Ok(false)
            }
            
            RetrainStep::Complete => {
                // Swap models
                if let Some(new_model) = self.pending_model.take() {
                    self.model = new_model;
                }
                
                self.is_retraining = false;
                self.changes_since_update = 0;
                self.retrain_progress = 1.0;
                self.retrain_step = RetrainStep::Idle;
                Ok(true)
            }
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn cancel_retrain(&mut self) -> Result<(), JsValue> {
        self.is_retraining = false;
        self.retrain_progress = 0.0;
        self.retrain_step = RetrainStep::Idle;
        self.pending_model = None;
        Ok(())
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn is_retraining(&self) -> bool {
        self.is_retraining
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn get_retrain_progress(&self) -> f32 {
        self.retrain_progress
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn export_model(&self) -> Result<String, JsValue> {
        serde_json::to_string(self)
            .map_err(|e| {
                #[cfg(target_arch = "wasm32")]
                return JsValue::from_str(&format!("Failed to export model: {}", e));
                #[cfg(not(target_arch = "wasm32"))]
                return format!("Failed to export model: {}", e);
            })
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn import_model(json_data: &str) -> Result<IncrementalEmbedder, JsValue> {
        serde_json::from_str(json_data)
            .map_err(|e| {
                #[cfg(target_arch = "wasm32")]
                return JsValue::from_str(&format!("Failed to import model: {}", e));
                #[cfg(not(target_arch = "wasm32"))]
                return format!("Failed to import model: {}", e);
            })
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn get_similarity(&self, text1: &str, text2: &str) -> Result<f32, JsValue> {
        let vec1 = self.transform(text1)?;
        let vec2 = self.transform(text2)?;
        Ok(cosine_similarity(&vec1, &vec2))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn get_document_count(&self) -> usize {
        self.documents.len()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn get_vocab_size(&self) -> usize {
        self.model.vocab_size()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn get_embedding_dim(&self) -> usize {
        self.model.embedding_dim()
    }
}

// Non-WASM methods for internal use
impl IncrementalEmbedder {
    pub fn transform_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, JsValue> {
        texts.iter()
            .map(|text| self.transform(text))
            .collect()
    }

    pub fn get_similarity_batch(&self, query: &str, candidates: Vec<String>) -> Result<Vec<f32>, JsValue> {
        let query_vec = self.transform(query)?;
        
        candidates.iter()
            .map(|candidate| {
                let candidate_vec = self.transform(candidate)?;
                Ok(cosine_similarity(&query_vec, &candidate_vec))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_incremental_embedder_basic() {
        let mut embedder = IncrementalEmbedder::new(0.5); // Higher threshold to avoid auto-retrain
        
        // Add documents
        embedder.add_document("今日は天気がいいですね".to_string(), 64).unwrap();
        embedder.add_document("明日は雨が降りそうです".to_string(), 64).unwrap();
        embedder.add_document("今日は映画を見ました".to_string(), 64).unwrap();
        
        // Transform a document
        let embedding = embedder.transform("今日は晴れです").unwrap();
        assert_eq!(embedding.len(), 64);
        
        // Check similarity
        let sim = embedder.get_similarity("今日は天気がいい", "明日は天気がいい").unwrap();
        assert!(sim >= -1.0 && sim <= 1.0);
    }

    #[test]
    fn test_background_retrain() {
        let mut embedder = IncrementalEmbedder::new(2.0); // Extremely high threshold to avoid auto-retrain
        
        // Add documents
        for i in 0..5 {
            embedder.add_document(format!("文書番号{}", i), 32).unwrap();
        }
        
        // Ensure no auto-retrain is in progress
        assert!(!embedder.is_retraining());
        
        // Start retraining
        embedder.start_background_retrain(32).unwrap();
        assert!(embedder.is_retraining());
        
        // Step through retraining
        let mut steps = 0;
        while !embedder.step_retrain().unwrap() && steps < 10 {
            steps += 1;
        }
        
        assert!(!embedder.is_retraining());
        assert_eq!(embedder.get_retrain_progress(), 1.0);
    }

    #[test]
    fn test_model_serialization() {
        let mut embedder = IncrementalEmbedder::new(0.3);
        embedder.add_document("テスト文書".to_string(), 32).unwrap();
        
        // Export model
        let json = embedder.export_model().unwrap();
        
        // Import model
        let restored = IncrementalEmbedder::import_model(&json).unwrap();
        
        assert_eq!(embedder.get_document_count(), restored.get_document_count());
    }
}