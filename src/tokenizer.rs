use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JapaneseTokenizer {
    min_ngram: usize,
    max_ngram: usize,
    min_doc_freq: usize,
    max_doc_freq_ratio: f32,
    max_vocab_size: usize,
}

impl Default for JapaneseTokenizer {
    fn default() -> Self {
        Self {
            min_ngram: 2,
            max_ngram: 3,
            min_doc_freq: 2,
            max_doc_freq_ratio: 0.8,
            max_vocab_size: 50000,
        }
    }
}

impl JapaneseTokenizer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn new_with_ngrams(min_ngram: usize, max_ngram: usize) -> Self {
        Self {
            min_ngram,
            max_ngram,
            ..Self::default()
        }
    }

    // Generate character n-grams from text
    pub fn char_ngrams(&self, text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
        let mut ngrams = Vec::new();

        for n in self.min_ngram..=self.max_ngram {
            if chars.len() >= n {
                for i in 0..=chars.len() - n {
                    let ngram: String = chars[i..i + n].iter().collect();
                    ngrams.push(ngram);
                }
            }
        }

        ngrams
    }

    // Extract continuous sequences of same character type
    pub fn char_type_sequences(&self, text: &str) -> Vec<String> {
        let mut sequences = Vec::new();
        let mut current_seq = String::new();
        let mut current_type = CharType::Other;

        for ch in text.chars() {
            let char_type = CharType::from_char(ch);
            
            if char_type != current_type && !current_seq.is_empty() {
                if current_type != CharType::Other && current_seq.len() > 1 {
                    sequences.push(current_seq.clone());
                }
                current_seq.clear();
            }

            if char_type != CharType::Other {
                current_seq.push(ch);
                current_type = char_type;
            }
        }

        if !current_seq.is_empty() && current_type != CharType::Other && current_seq.len() > 1 {
            sequences.push(current_seq);
        }

        sequences
    }

    // Simple word boundary estimation
    pub fn estimate_word_boundaries(&self, text: &str) -> Vec<String> {
        let mut words = Vec::new();
        let mut current_word = String::new();
        let mut prev_type = CharType::Other;

        for ch in text.chars() {
            let char_type = CharType::from_char(ch);

            // Detect boundaries
            let is_boundary = match (prev_type, char_type) {
                (CharType::Hiragana, CharType::Kanji) => true,
                (CharType::Katakana, CharType::Kanji) => true,
                (CharType::Kanji, CharType::Hiragana) => {
                    // Common particles following kanji
                    matches!(ch, 'を' | 'は' | 'が' | 'に' | 'で' | 'と' | 'の' | 'へ' | 'や')
                }
                (_, CharType::Other) | (CharType::Other, _) => true,
                _ => false,
            };

            if is_boundary && !current_word.is_empty() {
                if current_word.len() > 1 {
                    words.push(current_word.clone());
                }
                current_word.clear();
            }

            if char_type != CharType::Other {
                current_word.push(ch);
                prev_type = char_type;
            }
        }

        if !current_word.is_empty() && current_word.len() > 1 {
            words.push(current_word);
        }

        words
    }

    // Main tokenization function combining all methods
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = HashSet::new();

        // Add character n-grams
        for token in self.char_ngrams(text) {
            tokens.insert(token);
        }

        // Add character type sequences
        for token in self.char_type_sequences(text) {
            tokens.insert(token);
        }

        // Add estimated word boundaries
        for token in self.estimate_word_boundaries(text) {
            tokens.insert(token);
        }

        tokens.into_iter().collect()
    }

    // Build vocabulary from multiple documents
    pub fn build_vocabulary(&self, documents: &[String]) -> HashMap<String, usize> {
        let mut doc_freq: HashMap<String, usize> = HashMap::new();
        
        for doc in documents {
            let tokens: HashSet<String> = self.tokenize(doc).into_iter().collect();
            for token in tokens {
                *doc_freq.entry(token).or_insert(0) += 1;
            }
        }

        // Filter by document frequency
        let total_docs = documents.len();
        let max_docs = (total_docs as f32 * self.max_doc_freq_ratio) as usize;
        
        let mut filtered_vocab: Vec<(String, usize)> = doc_freq
            .into_iter()
            .filter(|(_, freq)| *freq >= self.min_doc_freq && *freq <= max_docs)
            .collect();

        // Sort by frequency and limit vocabulary size
        filtered_vocab.sort_by(|a, b| b.1.cmp(&a.1));
        filtered_vocab.truncate(self.max_vocab_size);

        // Create token to index mapping
        let mut vocab = HashMap::new();
        for (idx, (token, _)) in filtered_vocab.into_iter().enumerate() {
            vocab.insert(token, idx);
        }

        vocab
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum CharType {
    Hiragana,
    Katakana,
    Kanji,
    Alphabet,
    Number,
    Other,
}

impl CharType {
    fn from_char(ch: char) -> Self {
        match ch {
            'ぁ'..='ん' => CharType::Hiragana,
            'ァ'..='ヴ' | 'ー' => CharType::Katakana,
            '一'..='龯' => CharType::Kanji,
            'a'..='z' | 'A'..='Z' => CharType::Alphabet,
            '0'..='9' | '０'..='９' => CharType::Number,
            _ => CharType::Other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_char_ngrams() {
        let tokenizer = JapaneseTokenizer::new_with_ngrams(2, 3);
        let text = "今日は";
        let ngrams = tokenizer.char_ngrams(text);
        
        assert!(ngrams.contains(&"今日".to_string()));
        assert!(ngrams.contains(&"日は".to_string()));
        assert!(ngrams.contains(&"今日は".to_string()));
    }

    #[test]
    fn test_char_type_sequences() {
        let tokenizer = JapaneseTokenizer::new();
        let text = "今日は映画を見ました";
        let sequences = tokenizer.char_type_sequences(text);
        
        assert!(sequences.contains(&"今日".to_string()));
        assert!(sequences.contains(&"映画".to_string()));
        assert!(sequences.contains(&"ました".to_string()));
    }

    #[test]
    fn test_estimate_word_boundaries() {
        let tokenizer = JapaneseTokenizer::new();
        let text = "今日は映画を見ました";
        let words = tokenizer.estimate_word_boundaries(text);
        
        // Should contain some reasonable word segments
        assert!(!words.is_empty());
    }

    #[test]
    fn test_tokenize() {
        let tokenizer = JapaneseTokenizer::new();
        let text = "今日は映画を見ました";
        let tokens = tokenizer.tokenize(text);
        
        // Should generate multiple tokens
        assert!(tokens.len() > 5);
        
        // Should contain various n-grams
        assert!(tokens.contains(&"今日".to_string()));
        assert!(tokens.contains(&"映画".to_string()));
    }

    #[test]
    fn test_build_vocabulary() {
        let mut tokenizer = JapaneseTokenizer::new();
        tokenizer.min_doc_freq = 1; // Lower threshold for testing
        
        let documents = vec![
            "今日は天気がいいですね".to_string(),
            "明日は雨が降りそうです".to_string(),
            "今日は映画を見ました".to_string(),
            "天気は晴れです".to_string(),
            "映画は面白かったです".to_string(),
        ];
        
        let vocab = tokenizer.build_vocabulary(&documents);
        
        // Should contain common tokens
        assert!(vocab.contains_key("今日") || vocab.contains_key("天気") || vocab.contains_key("映画"));
        // Should have reasonable vocabulary size
        assert!(vocab.len() > 5);
        assert!(vocab.len() < 1000);
    }
}