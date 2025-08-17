use crate::utils::l2_normalize;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
#[derive(Debug, Clone)]
pub struct StableHashEmbedder {
    dimension: usize,
    char_ngram_size: usize,
    seed: u64,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl StableHashEmbedder {
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(constructor))]
    pub fn new(dimension: usize, char_ngram_size: usize) -> Self {
        Self {
            dimension,
            char_ngram_size,
            seed: 42, // Fixed seed for stability
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn new_with_seed(dimension: usize, char_ngram_size: usize, seed: u64) -> Self {
        Self {
            dimension,
            char_ngram_size,
            seed,
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn transform(&self, text: &str) -> Vec<f32> {
        let mut embedding = vec![0.0f32; self.dimension];
        
        // Generate character n-grams
        let chars: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
        
        if chars.len() < self.char_ngram_size {
            // Handle short texts
            self.hash_and_accumulate(&text, &mut embedding);
        } else {
            // Generate n-grams
            for i in 0..=chars.len() - self.char_ngram_size {
                let ngram: String = chars[i..i + self.char_ngram_size].iter().collect();
                self.hash_and_accumulate(&ngram, &mut embedding);
            }
        }
        
        // Add character type features
        self.add_char_type_features(text, &mut embedding);
        
        // Normalize the embedding
        l2_normalize(&mut embedding);
        
        embedding
    }

    fn hash_and_accumulate(&self, token: &str, embedding: &mut [f32]) {
        // Use multiple hash functions for better distribution
        for hash_idx in 0..3 {
            let hash_value = self.hash_token(token, hash_idx);
            let index = (hash_value as usize) % self.dimension;
            
            // Use hash value to determine sign (feature hashing trick)
            let sign = if hash_value & 1 == 0 { 1.0 } else { -1.0 };
            embedding[index] += sign;
        }
    }

    fn hash_token(&self, token: &str, hash_idx: u32) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.seed.hash(&mut hasher);
        hash_idx.hash(&mut hasher);
        token.hash(&mut hasher);
        hasher.finish()
    }

    fn add_char_type_features(&self, text: &str, embedding: &mut [f32]) {
        let mut hiragana_count = 0;
        let mut katakana_count = 0;
        let mut kanji_count = 0;
        let mut alphabet_count = 0;
        let mut number_count = 0;
        
        for ch in text.chars() {
            match ch {
                'ぁ'..='ん' => hiragana_count += 1,
                'ァ'..='ヴ' | 'ー' => katakana_count += 1,
                '一'..='龯' => kanji_count += 1,
                'a'..='z' | 'A'..='Z' => alphabet_count += 1,
                '0'..='9' | '０'..='９' => number_count += 1,
                _ => {}
            }
        }
        
        let total = text.len() as f32;
        if total > 0.0 {
            // Use last few dimensions for character type ratios
            let feature_start = self.dimension.saturating_sub(5);
            
            if feature_start < self.dimension {
                embedding[feature_start] = hiragana_count as f32 / total;
            }
            if feature_start + 1 < self.dimension {
                embedding[feature_start + 1] = katakana_count as f32 / total;
            }
            if feature_start + 2 < self.dimension {
                embedding[feature_start + 2] = kanji_count as f32 / total;
            }
            if feature_start + 3 < self.dimension {
                embedding[feature_start + 3] = alphabet_count as f32 / total;
            }
            if feature_start + 4 < self.dimension {
                embedding[feature_start + 4] = number_count as f32 / total;
            }
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn get_similarity(&self, text1: &str, text2: &str) -> f32 {
        let vec1 = self.transform(text1);
        let vec2 = self.transform(text2);
        crate::utils::cosine_similarity(&vec1, &vec2)
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn get_dimension(&self) -> usize {
        self.dimension
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn get_ngram_size(&self) -> usize {
        self.char_ngram_size
    }
}

// Non-WASM methods for internal use
impl StableHashEmbedder {
    pub fn transform_batch(&self, texts: Vec<String>) -> Vec<Vec<f32>> {
        texts.iter()
            .map(|text| self.transform(text))
            .collect()
    }

    pub fn get_similarity_batch(&self, query: &str, candidates: Vec<String>) -> Vec<f32> {
        let query_vec = self.transform(query);
        
        candidates.iter()
            .map(|candidate| {
                let candidate_vec = self.transform(candidate);
                crate::utils::cosine_similarity(&query_vec, &candidate_vec)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stable_hash_embedder_basic() {
        let embedder = StableHashEmbedder::new(64, 2);
        
        // Transform text
        let embedding = embedder.transform("今日は天気がいいですね");
        assert_eq!(embedding.len(), 64);
        
        // Check that embedding is not all zeros
        let sum: f32 = embedding.iter().map(|x| x.abs()).sum();
        assert!(sum > 0.0);
        
        // Check normalization
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_stability() {
        let embedder1 = StableHashEmbedder::new(32, 2);
        let embedder2 = StableHashEmbedder::new(32, 2);
        
        let text = "同じテキスト";
        
        let embedding1 = embedder1.transform(text);
        let embedding2 = embedder2.transform(text);
        
        // Should produce identical embeddings
        for (a, b) in embedding1.iter().zip(embedding2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_similarity() {
        let embedder = StableHashEmbedder::new(64, 2);
        
        // Similar texts should have high similarity
        let sim1 = embedder.get_similarity("今日は天気がいい", "今日は天気が良い");
        assert!(sim1 > 0.5);
        
        // Different texts should have lower similarity
        let sim2 = embedder.get_similarity("今日は天気がいい", "昨日は雨でした");
        assert!(sim2 < sim1);
    }

    #[test]
    fn test_short_text() {
        let embedder = StableHashEmbedder::new(32, 3);
        
        // Should handle text shorter than n-gram size
        let embedding = embedder.transform("あ");
        assert_eq!(embedding.len(), 32);
        
        let sum: f32 = embedding.iter().map(|x| x.abs()).sum();
        assert!(sum > 0.0);
    }

    #[test]
    fn test_different_seeds() {
        let embedder1 = StableHashEmbedder::new_with_seed(32, 2, 42);
        let embedder2 = StableHashEmbedder::new_with_seed(32, 2, 123);
        
        let text = "テストテキスト";
        
        let embedding1 = embedder1.transform(text);
        let embedding2 = embedder2.transform(text);
        
        // Different seeds should produce different embeddings
        let mut different = false;
        for (a, b) in embedding1.iter().zip(embedding2.iter()) {
            if (a - b).abs() > 1e-6 {
                different = true;
                break;
            }
        }
        assert!(different);
    }
}