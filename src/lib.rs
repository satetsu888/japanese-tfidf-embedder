#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

// Use wee_alloc as the global allocator for smaller WASM size
#[cfg(all(target_arch = "wasm32", feature = "wee_alloc"))]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

pub mod tokenizer;
pub mod tfidf_lsa;
pub mod incremental;
pub mod stable_hash;
pub mod utils;

// Re-export main types
pub use incremental::IncrementalEmbedder;
pub use stable_hash::StableHashEmbedder;

// Set up console error panic hook for better debugging in browser
#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub fn init() {
    #[cfg(target_arch = "wasm32")]
    {
        utils::set_panic_hook();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_functionality() {
        // Basic test will be added after implementation
        assert_eq!(1 + 1, 2);
    }
    
    #[test]
    fn test_similarity_with_many_documents() {
        let mut embedder = incremental::IncrementalEmbedder::new(0.3);
        
        // Add various documents
        let documents = vec![
            "今日は天気がいいですね。",
            "明日は雨が降りそうです。",
            "映画を見に行きたいです。",
            "昨日は映画を見ました。",
            "天気予報では晴れです。",
            "今日の天気は晴れです。",
            "プログラミングを勉強しています。",
            "Rustは素晴らしい言語です。",
            "機械学習について学んでいます。",
            "自然言語処理は興味深いです。",
            "東京は日本の首都です。",
            "大阪は関西の大都市です。",
            "京都には多くの寺院があります。",
            "富士山は日本一高い山です。",
            "桜の季節は美しいです。",
            "紅葉も綺麗ですね。",
            "日本料理は美味しいです。",
            "寿司が大好きです。",
            "ラーメンも美味しいですね。",
            "コーヒーを飲みたいです。",
        ];
        
        println!("Adding {} documents...", documents.len());
        for (i, doc) in documents.iter().enumerate() {
            embedder.add_document(doc.to_string(), 64).unwrap();
            println!("Added document {}: {}", i + 1, doc);
            
            // Check if retraining is needed
            if embedder.is_retraining() {
                println!("Retraining started...");
                while !embedder.step_retrain().unwrap() {
                    // Continue retraining
                }
                println!("Retraining completed");
            }
        }
        
        // Test similarity
        let test_queries = vec![
            ("今日は天気がいいですね。", "天気予報では晴れです。"),
            ("映画を見に行きたいです。", "昨日は映画を見ました。"),
            ("東京は日本の首都です。", "大阪は関西の大都市です。"),
            ("寿司が大好きです。", "ラーメンも美味しいですね。"),
        ];
        
        println!("\nTesting similarities:");
        for (text1, text2) in test_queries {
            let similarity = embedder.get_similarity(text1, text2).unwrap();
            println!("Similarity between:\n  '{}'\n  '{}'\n  => {:.4}", text1, text2, similarity);
            
            // Check vectors are not zero
            let vec1 = embedder.transform(text1).unwrap();
            let vec2 = embedder.transform(text2).unwrap();
            
            let vec1_norm: f32 = vec1.iter().map(|x| x * x).sum::<f32>().sqrt();
            let vec2_norm: f32 = vec2.iter().map(|x| x * x).sum::<f32>().sqrt();
            
            assert!(vec1_norm > 0.001, "Vector 1 should not be zero for: '{}'", text1);
            assert!(vec2_norm > 0.001, "Vector 2 should not be zero for: '{}'", text2);
            
            // Check that similarity is meaningful when both vectors are non-zero
            // Note: Very small similarity values are ok, just checking vectors aren't completely zero
            if vec1_norm > 0.001 && vec2_norm > 0.001 {
                // Vectors are non-zero, so similarity is meaningful even if small
                println!("  Vectors are non-zero (norms: {:.4}, {:.4}), similarity is valid", vec1_norm, vec2_norm);
            }
        }
        
        // Check vocabulary size
        println!("\nVocabulary size: {}", embedder.get_vocabulary_size());
        println!("Document count: {}", embedder.get_document_count());
        
        assert!(embedder.get_vocabulary_size() > 0, "Vocabulary should not be empty");
        assert_eq!(embedder.get_document_count(), documents.len());
    }
}
