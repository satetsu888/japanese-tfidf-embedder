use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DictionaryEntry {
    pub surface: String,
    pub variants: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserDictionary {
    entries: Vec<DictionaryEntry>,
    variant_to_surface: HashMap<String, String>,
}

impl UserDictionary {
    pub fn new(entries: Vec<DictionaryEntry>) -> Self {
        let mut variant_to_surface = HashMap::new();
        
        for entry in &entries {
            variant_to_surface.insert(entry.surface.clone(), entry.surface.clone());
            
            for variant in &entry.variants {
                variant_to_surface.insert(variant.clone(), entry.surface.clone());
            }
        }
        
        let mut dict = Self {
            entries,
            variant_to_surface,
        };
        
        dict.sort_entries_by_length();
        dict
    }
    
    fn sort_entries_by_length(&mut self) {
        for entry in &mut self.entries {
            entry.variants.sort_by_key(|v| std::cmp::Reverse(v.chars().count()));
        }
        
        self.entries.sort_by_key(|e| {
            let max_len = e.variants.iter()
                .map(|v| v.chars().count())
                .max()
                .unwrap_or(0)
                .max(e.surface.chars().count());
            std::cmp::Reverse(max_len)
        });
    }
    
    pub fn find_matches(&self, text: &str) -> Vec<(usize, usize, String)> {
        let mut matches = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let mut processed = vec![false; chars.len()];
        
        for i in 0..chars.len() {
            if processed[i] {
                continue;
            }
            
            for entry in &self.entries {
                let all_patterns: Vec<&str> = std::iter::once(entry.surface.as_str())
                    .chain(entry.variants.iter().map(|s| s.as_str()))
                    .collect();
                
                for pattern in all_patterns {
                    let pattern_chars: Vec<char> = pattern.chars().collect();
                    if i + pattern_chars.len() <= chars.len() {
                        let text_slice: String = chars[i..i + pattern_chars.len()].iter().collect();
                        if text_slice == pattern {
                            let mut all_processed = true;
                            for j in i..i + pattern_chars.len() {
                                if processed[j] {
                                    all_processed = false;
                                    break;
                                }
                            }
                            
                            if all_processed {
                                matches.push((i, i + pattern_chars.len(), entry.surface.clone()));
                                for j in i..i + pattern_chars.len() {
                                    processed[j] = true;
                                }
                                break;
                            }
                        }
                    }
                }
            }
        }
        
        matches
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JapaneseTokenizer {
    min_ngram: usize,
    max_ngram: usize,
    min_doc_freq: usize,
    max_doc_freq_ratio: f32,
    max_vocab_size: usize,
    stop_words: HashSet<String>,
    enable_stop_words: bool,
    pub(crate) user_dictionary: Option<UserDictionary>,
}

impl Default for JapaneseTokenizer {
    fn default() -> Self {
        let mut tokenizer = Self {
            min_ngram: 2,
            max_ngram: 3,
            min_doc_freq: 1,  // Changed from 2 to 1 to avoid empty vocabulary
            max_doc_freq_ratio: 0.9,  // Increased from 0.8 to be less strict
            max_vocab_size: 50000,
            stop_words: HashSet::new(),
            enable_stop_words: true,
            user_dictionary: None,
        };
        tokenizer.initialize_stop_words();
        tokenizer
    }
}

impl JapaneseTokenizer {
    pub fn new() -> Self {
        Self::default()
    }

    fn initialize_stop_words(&mut self) {
        // Japanese particles (助詞)
        let particles = vec![
            "は", "が", "を", "に", "で", "と", "の", "へ", "や", "から", 
            "まで", "より", "など", "ば", "も", "か", "し", "ね", "よ", "わ",
            "ぞ", "ぜ", "さ", "な", "だけ", "でも", "しか", "ほど", "くらい", "ばかり"
        ];
        
        // Auxiliary verbs (助動詞)
        let auxiliaries = vec![
            "です", "ます", "だ", "である", "でした", "ました", "でしょう", "ましょう",
            "だろう", "であろう", "かもしれない", "かもしれません", "ない", "ません", "なかった", "ませんでした"
        ];
        
        // Formal nouns (形式名詞)
        let formal_nouns = vec![
            "こと", "もの", "ため", "よう", "はず", "つもり", "わけ", "ところ", "ほう"
        ];
        
        // Conjunctions (接続詞)
        let conjunctions = vec![
            "また", "しかし", "そして", "それで", "だから", "つまり", "ただし", "なお", "および", "または"
        ];
        
        // Common suffixes and prefixes
        let affixes = vec![
            "お", "ご", "御", "的", "性", "化", "者", "たち", "ら", "ども"
        ];
        
        // Add all stop words to the set
        for word in particles.iter()
            .chain(auxiliaries.iter())
            .chain(formal_nouns.iter())
            .chain(conjunctions.iter())
            .chain(affixes.iter()) {
            self.stop_words.insert(word.to_string());
        }
    }

    pub fn new_with_ngrams(min_ngram: usize, max_ngram: usize) -> Self {
        Self {
            min_ngram,
            max_ngram,
            ..Self::default()
        }
    }
    
    pub fn set_user_dictionary(&mut self, entries: Vec<DictionaryEntry>) {
        self.user_dictionary = Some(UserDictionary::new(entries));
    }
    
    pub fn clear_user_dictionary(&mut self) {
        self.user_dictionary = None;
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

    // Extract single kanji characters (1-grams for kanji only)
    pub fn kanji_unigrams(&self, text: &str) -> Vec<String> {
        let mut unigrams = Vec::new();
        
        for ch in text.chars() {
            if matches!(CharType::from_char(ch), CharType::Kanji) {
                unigrams.push(ch.to_string());
            }
        }
        
        unigrams
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

        // If user dictionary is available, find matches first
        if let Some(ref dictionary) = self.user_dictionary {
            let matches = dictionary.find_matches(text);
            
            // Add dictionary matches as tokens
            for (_start, _end, surface) in &matches {
                tokens.insert(surface.clone());
            }
            
            // Process unmatched portions with regular tokenization
            let chars: Vec<char> = text.chars().collect();
            let mut processed = vec![false; chars.len()];
            
            // Mark matched regions as processed
            for (start, end, _) in &matches {
                for i in *start..*end {
                    processed[i] = true;
                }
            }
            
            // Extract unmatched segments
            let mut segments = Vec::new();
            let mut current_segment = String::new();
            
            for (i, ch) in chars.iter().enumerate() {
                if !processed[i] {
                    current_segment.push(*ch);
                } else if !current_segment.is_empty() {
                    segments.push(current_segment.clone());
                    current_segment.clear();
                }
            }
            
            if !current_segment.is_empty() {
                segments.push(current_segment);
            }
            
            // Apply regular tokenization to unmatched segments
            for segment in segments {
                for token in self.char_ngrams(&segment) {
                    if !self.should_filter_token(&token) {
                        tokens.insert(token);
                    }
                }
                
                for token in self.kanji_unigrams(&segment) {
                    if !self.should_filter_token(&token) {
                        tokens.insert(token);
                    }
                }
                
                for token in self.char_type_sequences(&segment) {
                    if !self.should_filter_token(&token) {
                        tokens.insert(token);
                    }
                }
                
                for token in self.estimate_word_boundaries(&segment) {
                    if !self.should_filter_token(&token) {
                        tokens.insert(token);
                    }
                }
            }
        } else {
            // No dictionary, use regular tokenization
            for token in self.char_ngrams(text) {
                if !self.should_filter_token(&token) {
                    tokens.insert(token);
                }
            }
            
            for token in self.kanji_unigrams(text) {
                if !self.should_filter_token(&token) {
                    tokens.insert(token);
                }
            }

            for token in self.char_type_sequences(text) {
                if !self.should_filter_token(&token) {
                    tokens.insert(token);
                }
            }

            for token in self.estimate_word_boundaries(text) {
                if !self.should_filter_token(&token) {
                    tokens.insert(token);
                }
            }
        }

        tokens.into_iter().collect()
    }

    // Check if a token should be filtered
    fn should_filter_token(&self, token: &str) -> bool {
        if !self.enable_stop_words {
            return false;
        }
        
        // Filter exact stop words
        if self.stop_words.contains(token) {
            return true;
        }
        
        // Filter tokens that are only stop words
        // (e.g., "です" should be filtered, but "ですね" might be kept)
        if token.len() <= 3 && self.stop_words.contains(token) {
            return true;
        }
        
        false
    }

    // Calculate token quality score (for N-gram quality scoring)
    pub fn calculate_token_score(&self, token: &str, doc_freq: usize, total_docs: usize) -> f32 {
        let mut score = 1.0;
        
        // Check if token is a dictionary word (high priority)
        if let Some(ref dictionary) = self.user_dictionary {
            if dictionary.variant_to_surface.contains_key(token) {
                score *= 2.0;  // Boost score for dictionary words
            }
        }
        
        // Check if token is a single kanji (1-gram)
        let chars: Vec<char> = token.chars().collect();
        if chars.len() == 1 && matches!(CharType::from_char(chars[0]), CharType::Kanji) {
            // Single kanji: reduce weight since same kanji can have different meanings in different contexts
            score *= 0.6;  // Lower weight for single kanji
        }
        
        // Reduce score for tokens starting/ending with particles
        let particles = ["は", "が", "を", "に", "で", "と", "の", "へ"];
        for particle in particles.iter() {
            if token.starts_with(particle) || token.ends_with(particle) {
                score *= 0.5;
            }
        }
        
        // Check character type consistency
        let has_kanji = token.chars().any(|c| matches!(CharType::from_char(c), CharType::Kanji));
        let has_hiragana = token.chars().any(|c| matches!(CharType::from_char(c), CharType::Hiragana));
        let has_katakana = token.chars().any(|c| matches!(CharType::from_char(c), CharType::Katakana));
        
        let char_type_count = (has_kanji as u8) + (has_hiragana as u8) + (has_katakana as u8);
        
        // Boost score for tokens with single character type (more cohesive)
        // But skip this boost for single kanji (already handled above)
        if char_type_count == 1 && chars.len() > 1 {
            score *= 1.5;  // Single character type = likely a complete word
        } else if char_type_count >= 2 {
            score *= 0.7;  // Mixed character types = likely fragmented
        }
        
        // Additional boost for pure kanji or katakana compounds (multi-character, meaningful words)
        if char_type_count == 1 && chars.len() > 1 && (has_kanji || has_katakana) {
            score *= 1.2;
        }
        
        // TF-IDF inspired scoring
        let idf = (total_docs as f32 / doc_freq as f32).ln();
        score *= idf;
        
        score
    }

    // Build vocabulary from multiple documents with quality scoring
    pub fn build_vocabulary(&self, documents: &[String]) -> HashMap<String, usize> {
        let mut doc_freq: HashMap<String, usize> = HashMap::new();
        
        for doc in documents {
            let tokens: HashSet<String> = self.tokenize(doc).into_iter().collect();
            for token in tokens {
                *doc_freq.entry(token).or_insert(0) += 1;
            }
        }

        let total_docs = documents.len();
        let max_docs = ((total_docs as f32 * self.max_doc_freq_ratio) as usize).max(1);
        
        // Filter and score tokens
        let mut scored_vocab: Vec<(String, f32)> = doc_freq
            .iter()
            .filter(|(_, freq)| **freq >= self.min_doc_freq && **freq <= max_docs)
            .map(|(token, freq)| {
                let score = self.calculate_token_score(token, *freq, total_docs);
                (token.clone(), score)
            })
            .collect();

        // Sort by quality score instead of just frequency
        scored_vocab.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Dynamic vocabulary size based on document count
        let dynamic_vocab_size = self.calculate_dynamic_vocab_size(total_docs);
        scored_vocab.truncate(dynamic_vocab_size);

        // Create token to index mapping
        let mut vocab = HashMap::new();
        for (idx, (token, _)) in scored_vocab.into_iter().enumerate() {
            vocab.insert(token, idx);
        }

        vocab
    }

    // Calculate dynamic vocabulary size based on document count
    fn calculate_dynamic_vocab_size(&self, doc_count: usize) -> usize {
        // Base size: 100 tokens per document, capped at max_vocab_size
        let base_size = doc_count * 100;
        let adjusted_size = if doc_count < 10 {
            base_size.max(1000)  // Minimum 1000 tokens for small collections
        } else if doc_count < 100 {
            base_size.max(5000)  // Minimum 5000 for medium collections
        } else {
            base_size.max(10000) // Minimum 10000 for large collections
        };
        
        adjusted_size.min(self.max_vocab_size)
    }

    // Setter methods for configuration
    pub fn set_stop_words_enabled(&mut self, enabled: bool) {
        self.enable_stop_words = enabled;
    }
    
    pub fn add_stop_word(&mut self, word: &str) {
        self.stop_words.insert(word.to_string());
    }
    
    pub fn remove_stop_word(&mut self, word: &str) {
        self.stop_words.remove(word);
    }
    
    pub fn get_stop_words(&self) -> &HashSet<String> {
        &self.stop_words
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
    fn test_kanji_unigrams() {
        let tokenizer = JapaneseTokenizer::new();
        let text = "今日は映画を見ました";
        let unigrams = tokenizer.kanji_unigrams(text);
        
        // Should contain individual kanji characters
        assert!(unigrams.contains(&"今".to_string()));
        assert!(unigrams.contains(&"日".to_string()));
        assert!(unigrams.contains(&"映".to_string()));
        assert!(unigrams.contains(&"画".to_string()));
        assert!(unigrams.contains(&"見".to_string()));
        
        // Should not contain hiragana
        assert!(!unigrams.contains(&"は".to_string()));
        assert!(!unigrams.contains(&"を".to_string()));
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
        
        // Should also contain kanji unigrams
        assert!(tokens.contains(&"今".to_string()), "Should contain single kanji '今'");
        assert!(tokens.contains(&"日".to_string()), "Should contain single kanji '日'");
        assert!(tokens.contains(&"映".to_string()), "Should contain single kanji '映'");
        assert!(tokens.contains(&"画".to_string()), "Should contain single kanji '画'");
        assert!(tokens.contains(&"見".to_string()), "Should contain single kanji '見'");
    }

    #[test]
    fn test_stop_words() {
        let tokenizer = JapaneseTokenizer::new();
        
        // Check that stop words are initialized
        assert!(!tokenizer.get_stop_words().is_empty());
        assert!(tokenizer.get_stop_words().contains("は"));
        assert!(tokenizer.get_stop_words().contains("です"));
        assert!(tokenizer.get_stop_words().contains("こと"));
        
        // Test tokenization with stop words
        let text = "今日は天気です";
        let tokens = tokenizer.tokenize(text);
        
        // Should not contain isolated stop words
        assert!(!tokens.contains(&"は".to_string()));
        assert!(!tokens.contains(&"です".to_string()));
        
        // But should contain meaningful tokens
        assert!(tokens.iter().any(|t| t.contains("今日")));
        assert!(tokens.iter().any(|t| t.contains("天気")));
    }

    #[test]
    fn test_token_quality_scoring() {
        let tokenizer = JapaneseTokenizer::new();
        
        // Token with particle should have lower score
        let particle_token_score = tokenizer.calculate_token_score("映画を", 5, 10);
        let normal_token_score = tokenizer.calculate_token_score("映画", 5, 10);
        assert!(normal_token_score > particle_token_score);
        
        // Token with single character type should have higher score than mixed
        let single_kanji_score = tokenizer.calculate_token_score("映画", 5, 10);
        let mixed_score = tokenizer.calculate_token_score("映画を", 5, 10);
        assert!(single_kanji_score > mixed_score);
        
        // Pure hiragana should have lower score than pure kanji/katakana
        let kanji_score = tokenizer.calculate_token_score("映画", 5, 10);
        let hiragana_score = tokenizer.calculate_token_score("えいが", 5, 10);
        assert!(kanji_score > hiragana_score);
        
        // Single kanji should have lower score than kanji compound
        let single_kanji_score = tokenizer.calculate_token_score("映", 5, 10);
        let compound_kanji_score = tokenizer.calculate_token_score("映画", 5, 10);
        assert!(compound_kanji_score > single_kanji_score, 
                "Compound kanji '映画' should have higher score than single kanji '映'");
    }

    #[test]
    fn test_dynamic_vocab_size() {
        let tokenizer = JapaneseTokenizer::new();
        
        // Small collection
        let small_size = tokenizer.calculate_dynamic_vocab_size(5);
        assert!(small_size >= 1000);
        
        // Medium collection
        let medium_size = tokenizer.calculate_dynamic_vocab_size(50);
        assert!(medium_size >= 5000);
        
        // Large collection
        let large_size = tokenizer.calculate_dynamic_vocab_size(200);
        assert!(large_size >= 10000);
        assert!(large_size <= 50000); // Should not exceed max_vocab_size
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
        
        // Should not contain isolated stop words
        assert!(!vocab.contains_key("は"));
        assert!(!vocab.contains_key("です"));
    }

    #[test]
    fn test_stop_words_configuration() {
        let mut tokenizer = JapaneseTokenizer::new();
        
        // Test disabling stop words
        tokenizer.set_stop_words_enabled(false);
        let text = "今日は天気です";
        let _tokens = tokenizer.tokenize(text);
        
        // With stop words disabled, particles might appear in tokens
        // (depending on n-gram generation)
        
        // Test adding custom stop word
        tokenizer.set_stop_words_enabled(true);
        tokenizer.add_stop_word("天気");
        assert!(tokenizer.get_stop_words().contains("天気"));
        
        // Test removing stop word
        tokenizer.remove_stop_word("は");
        assert!(!tokenizer.get_stop_words().contains("は"));
    }
    
    #[test]
    fn test_user_dictionary() {
        let mut tokenizer = JapaneseTokenizer::new();
        
        // Create dictionary entries
        let entries = vec![
            DictionaryEntry {
                surface: "人工知能".to_string(),
                variants: vec!["AI".to_string(), "エーアイ".to_string(), "Artificial Intelligence".to_string()],
            },
            DictionaryEntry {
                surface: "機械学習".to_string(),
                variants: vec!["ML".to_string(), "マシンラーニング".to_string()],
            },
        ];
        
        tokenizer.set_user_dictionary(entries);
        
        // Test with variants
        let text1 = "AIとMLの研究をしています";
        let tokens1 = tokenizer.tokenize(text1);
        assert!(tokens1.contains(&"人工知能".to_string()), "AI should be normalized to 人工知能");
        assert!(tokens1.contains(&"機械学習".to_string()), "ML should be normalized to 機械学習");
        
        // Test with surface forms
        let text2 = "人工知能と機械学習の研究";
        let tokens2 = tokenizer.tokenize(text2);
        assert!(tokens2.contains(&"人工知能".to_string()));
        assert!(tokens2.contains(&"機械学習".to_string()));
        
        // Test with katakana variants
        let text3 = "エーアイとマシンラーニング";
        let tokens3 = tokenizer.tokenize(text3);
        assert!(tokens3.contains(&"人工知能".to_string()), "エーアイ should be normalized to 人工知能");
        assert!(tokens3.contains(&"機械学習".to_string()), "マシンラーニング should be normalized to 機械学習");
        
        // Test clear dictionary
        tokenizer.clear_user_dictionary();
        let text4 = "AIとML";
        let tokens4 = tokenizer.tokenize(text4);
        assert!(!tokens4.contains(&"人工知能".to_string()), "After clearing, AI should not be normalized");
    }
    
    #[test]
    fn test_dictionary_score_boost() {
        let mut tokenizer = JapaneseTokenizer::new();
        
        let entries = vec![
            DictionaryEntry {
                surface: "人工知能".to_string(),
                variants: vec!["AI".to_string()],
            },
        ];
        
        tokenizer.set_user_dictionary(entries);
        
        // Dictionary word should have higher score
        let dict_score = tokenizer.calculate_token_score("人工知能", 5, 10);
        let normal_score = tokenizer.calculate_token_score("普通単語", 5, 10);
        
        assert!(dict_score > normal_score, "Dictionary words should have higher scores");
    }
}