# 日本語テキストベクトル化ライブラリ

WASM対応の軽量な日本語テキストベクトル化ライブラリです。TF-IDF + LSA（潜在意味解析）を使用し、段階的学習に対応しています。

## 特徴

- 🚀 **高速・軽量**: WASM対応でブラウザ上で動作（2-5MB以下）
- 🇯🇵 **日本語特化**: 文字N-gram、文字種別トークナイザーによる日本語最適化
- 📈 **段階的学習**: 文書を追加しながら動的にモデルを更新
- 🔍 **類似検索**: コサイン類似度による高速な類似文書検索
- 💾 **永続化対応**: モデルのJSON形式でのエクスポート/インポート
- 🔧 **2つの実装**:
  - `IncrementalEmbedder`: 段階的学習対応の動的ベクトル化
  - `StableHashEmbedder`: 文書追加に影響されない安定ベクトル化

## インストール

### 必要な環境

- Rust 1.70+
- wasm-pack
- Node.js 14+ (オプション)

### ビルド方法

```bash
# リポジトリのクローン
git clone https://github.com/yourusername/japanese-text-vector.git
cd japanese-text-vector

# 依存関係のインストール
cargo build

# WASMビルド（ブラウザ用）
wasm-pack build --target web --out-dir pkg

# WASMビルド（Node.js用）
wasm-pack build --target nodejs --out-dir pkg-node

# テストの実行
cargo test
```

## 使用方法

### ブラウザでの使用

```html
<!DOCTYPE html>
<html>
<head>
    <script type="module">
        import init, { IncrementalEmbedder, StableHashEmbedder } from './pkg/japanese_text_vector.js';

        async function run() {
            // WASMモジュールの初期化
            await init();

            // IncrementalEmbedder の使用
            const embedder = new IncrementalEmbedder(0.3);
            
            // 文書の追加
            embedder.add_document("今日は天気がいいですね", 64);
            embedder.add_document("明日は雨が降りそうです", 64);
            
            // ベクトル化
            const embedding = embedder.transform("今日は晴れです");
            console.log("Embedding:", embedding);
            
            // 類似度計算
            const similarity = embedder.get_similarity(
                "今日は天気がいい",
                "明日は天気がいい"
            );
            console.log("Similarity:", similarity);
        }

        run();
    </script>
</head>
</html>
```

### Node.jsでの使用

```javascript
const { IncrementalEmbedder, StableHashEmbedder } = require('./pkg-node/japanese_text_vector.js');

// StableHashEmbedder の使用（文書追加に影響されない）
const stableEmbedder = new StableHashEmbedder(64, 2);

const embedding = stableEmbedder.transform("日本語のテキスト");
console.log("Stable embedding:", embedding);
```

## API リファレンス

### IncrementalEmbedder

段階的学習に対応したテキストベクトル化クラス。

#### コンストラクタ

```javascript
new IncrementalEmbedder(update_threshold)
```

- `update_threshold`: 自動再学習のしきい値（0.0-1.0）

#### メソッド

| メソッド | 説明 |
|---------|------|
| `add_document(text, embedding_dim)` | 文書を追加 |
| `transform(text)` | テキストをベクトル化 |
| `get_similarity(text1, text2)` | 2つのテキストの類似度を計算 |
| `start_background_retrain(embedding_dim)` | バックグラウンド再学習を開始 |
| `step_retrain()` | 再学習を1ステップ実行 |
| `is_retraining()` | 再学習中かどうか |
| `get_retrain_progress()` | 再学習の進捗（0.0-1.0） |
| `export_model()` | モデルをJSON形式でエクスポート |
| `import_model(json_data)` | JSONからモデルを復元 |

### StableHashEmbedder

ハッシュベースの安定したベクトル化クラス。

#### コンストラクタ

```javascript
new StableHashEmbedder(dimension, char_ngram_size)
```

- `dimension`: ベクトルの次元数
- `char_ngram_size`: 文字N-gramのサイズ

#### メソッド

| メソッド | 説明 |
|---------|------|
| `transform(text)` | テキストをベクトル化 |
| `get_similarity(text1, text2)` | 2つのテキストの類似度を計算 |

## デモ

`examples/` ディレクトリにデモページが含まれています：

1. **basic_usage.html**: 基本的な使用例
2. **incremental_demo.html**: 段階的学習の対話的デモ

デモを実行するには：

```bash
# HTTPサーバーを起動
python3 -m http.server 8000

# ブラウザでアクセス
# http://localhost:8000/examples/basic_usage.html
```

## パフォーマンス

| 指標 | 目標値 | 実測値 |
|------|--------|--------|
| ベクトル化速度 | < 10ms/文書 | ~5ms |
| モデルサイズ | < 5MB | ~2MB |
| 語彙サイズ | < 50,000語 | 動的 |
| 埋め込み次元 | 64-128 | 設定可能 |

## 技術仕様

### 日本語トークナイザー

- **文字N-gram**: 2-gram, 3-gram の組み合わせ
- **文字種別抽出**: ひらがな、カタカナ、漢字の連続抽出
- **単語境界推定**: 文字種変化による簡易的な単語分割

### TF-IDF + LSA

- **TF-IDF**: 文書頻度による重み付け
- **LSA**: SVD（特異値分解）による次元削減
- **正規化**: L2正規化されたベクトル出力

## ライセンス

MIT License

## 貢献

プルリクエストを歓迎します！大きな変更の場合は、まずissueを開いて変更内容について議論してください。

## 今後の計画

- [ ] WebGPU対応による高速化
- [ ] より高度な日本語形態素解析の統合
- [ ] カスタマイズ可能なトークナイザー
- [ ] ストリーミング処理対応
- [ ] メモリ効率の更なる改善

## 謝辞

このプロジェクトは以下の技術を使用しています：

- [wasm-bindgen](https://github.com/rustwasm/wasm-bindgen)
- [nalgebra](https://nalgebra.org/)
- [serde](https://serde.rs/)