# Japanese Text Vector - Project Overview

## 🎯 Project Purpose

A lightweight WASM-compatible Japanese text vectorization library using TF-IDF + LSA (Latent Semantic Analysis) to calculate semantic similarity of Japanese documents in real-time within browsers.

## 🏗️ Architecture

### Core Technologies

- **Language**: Rust + WebAssembly (WASM)
- **Algorithm**: TF-IDF + LSA (dimensionality reduction via SVD)
- **Japanese Processing**: Character N-gram + character type detection for morphological analysis-free design

### Module Structure

```
src/
├── lib.rs              # WASM exports
├── tokenizer.rs        # Japanese tokenizer
├── tfidf_lsa.rs       # TF-IDF + LSA implementation
├── incremental.rs      # Incremental learning features
├── stable_hash.rs      # Hash-based stable vectorization
└── utils.rs           # Utility functions
```

## 💡 Key Features

### 1. IncrementalEmbedder (Incremental Learning Version)

- **Dynamic Model Updates**: Automatically retrains model as documents are added
- **Duplicate Detection**: Automatic exclusion of identical documents using HashSet
- **Background Processing**: Non-blocking retraining via step_retrain()
- **Persistence**: Model save/restore in JSON format

### 2. StableHashEmbedder (Stable Hash Version)

- **Stability**: Fixed vectorization unaffected by document additions
- **High Performance**: Instant vector generation using hash functions
- **Memory Efficient**: Lightweight implementation without learning models

### 3. Japanese Tokenizer

- **Character N-grams**: Combination of 2-grams and 3-grams
- **Character Type Detection**: Identification of hiragana, katakana, kanji, alphanumeric
- **Word Boundary Estimation**: Simple word segmentation based on character type changes
- **Vocabulary Limiting**: Document frequency filtering (max 50,000 words)

## 📊 Performance Metrics

| Metric | Achievement |
|--------|------------|
| WASM Size | **182KB** (with high-accuracy SVD) |
| Vectorization Speed | ~5ms/document |
| Embedding Dimensions | 64-128 (configurable) |
| Memory Usage | Minimal (using wee_alloc) |
| SVD Implementation | Full nalgebra SVD for high accuracy |

## 🎨 Demo Pages

### 1. basic_usage.html

- Basic usage examples
- Try both StableHashEmbedder and IncrementalEmbedder
- Similarity calculation and batch search features

### 2. incremental_demo.html

- **300 sample documents** (15 categories × 20 documents)
- Real-time incremental learning visualization
- Duplicate detection demonstration
- Performance metrics display

## 🔧 Technical Optimizations

### Memory Optimization

- Lightweight allocator using `wee_alloc`
- Size optimization with `opt-level = "s"`
- Efficient sparse matrix processing

### Japanese-Specific Design

- No morphological analysis required (no dictionary dependencies like MeCab)
- Semantic boundary estimation using character types
- Word segmentation through particle detection

### WASM Integration

- Conditional compilation (`#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]`)
- Support for testing on non-WASM targets
- Automatic JavaScript API generation

## 📈 Future Extensibility

### Implemented

- ✅ Basic TF-IDF + LSA
- ✅ Incremental learning functionality
- ✅ Automatic duplicate document detection
- ✅ Model persistence
- ✅ 300 sample documents

### Planned

- ⏳ WebGPU support for acceleration
- ⏳ Advanced dimensionality reduction methods (t-SNE, UMAP)
- ⏳ Customizable tokenizer
- ⏳ Streaming processing support

## 🚀 Build & Test

```bash
# Run Rust tests (all 19 tests passing)
cargo test

# Build WASM
wasm-pack build --target web --out-dir pkg

# Launch demo page
python3 -m http.server 8000
# Navigate to http://localhost:8000/examples/incremental_demo.html
```

## 📝 Usage Example

```javascript
// IncrementalEmbedder - Incremental learning version
const embedder = new IncrementalEmbedder(0.3);

// Add documents (duplicates automatically excluded)
embedder.add_document("今日は天気がいいですね", 64);
embedder.add_document("今日は天気がいいですね", 64); // Skipped

// Vectorization
const vector = embedder.transform("今日は晴れです");

// Calculate similarity
const similarity = embedder.get_similarity(
    "今日は天気がいい",
    "明日は天気が悪い"
);

// Check document existence
if (!embedder.contains_document(text)) {
    embedder.add_document(text, 64);
}
```

## 🎯 Project Strengths

1. **Lightweight**: Minimal browser load at just 152KB
2. **Fast**: Real-time vectorization and similarity calculation
3. **Japanese-Optimized**: Dictionary-free optimization for Japanese characteristics
4. **Flexible**: Two implementations - incremental learning and stable version
5. **Practical**: Rich features including duplicate detection, persistence, and background processing

## 📌 Limitations

- Simplified LSA implementation may reduce accuracy with large-scale data
- Document count limited by browser memory constraints
- Character-based processing makes homophone distinction difficult

## 🔍 Debugging Tips

- Check WASM errors in browser console
- Use `contains_document()` for duplicate checking
- Verify actual document count with `get_unique_document_count()`
- Monitor memory usage in Performance tab

## 🔬 Technical Details

### Why TF-IDF + LSA?

- **TF-IDF**: Captures term importance across document collection
- **LSA**: Reduces dimensionality while preserving semantic relationships
- **Combination**: Balances computational efficiency with semantic accuracy
- **High-Accuracy SVD**: Using nalgebra's full SVD implementation for precise latent semantic analysis
  - Singular value weighting for better representation
  - Preserves important latent dimensions
  - Improved similarity calculations

### Incremental Learning Strategy

1. **Threshold-based triggering**: Retrains when change ratio exceeds threshold
2. **Non-blocking execution**: Uses step-by-step processing to avoid UI freezing
3. **Model swapping**: Seamlessly switches to new model once training completes

### Japanese Language Handling

- **No dictionary required**: Uses character patterns instead of morphological analysis
- **Multi-granularity**: Combines different n-gram sizes for better coverage
- **Character type awareness**: Leverages Japanese script transitions for word boundaries

## 🌟 Unique Aspects

This project stands out by providing a fully client-side Japanese text analysis solution that:

- Runs entirely in the browser without server dependencies
- Handles Japanese text without external dictionaries
- Adapts to new documents dynamically
- Maintains sub-200KB footprint
- Provides both stable and adaptive vectorization strategies

The combination of these features makes it ideal for privacy-conscious applications, offline-capable PWAs, and real-time text analysis scenarios where server round-trips are undesirable.
