# japanese-tfidf-embedder

高精度SVD実装を搭載した、WASM対応の軽量な日本語テキストベクトル化ライブラリです。TF-IDF + LSA（潜在意味解析）を使用し、ブラウザ上でリアルタイムに日本語文書の意味的類似度を計算します。

### オンラインデモ

https://satetsu888.github.io/japanese-tfidf-embedder/

## 🎯 特徴

- 🚀 **高速・軽量**: WASM対応でブラウザ上で動作（184KB）
- 🇯🇵 **日本語特化**: 
  - 文字N-gram（2-3gram）+ 漢字1-gram
  - 日本語Stop-word対応（助詞・助動詞など約50語）
  - 文字種別認識による単語境界推定
- 📊 **高精度SVD**: nalgebraの完全SVD実装による正確な次元削減
- 📈 **段階的学習**: 文書を追加しながら動的にモデルを更新
- 🔍 **類似検索**: コサイン類似度による高速な類似文書検索
- 🚫 **重複排除**: 同一文書の自動検出と排除
- 💾 **永続化対応**: モデルのJSON形式でのエクスポート/インポート
- 🔧 **2つの実装**:
  - `IncrementalEmbedder`: 段階的学習対応の動的ベクトル化（重複検出機能付き）
  - `StableHashEmbedder`: 文書追加に影響されない安定ベクトル化
- 📚 **ユーザー辞書**: カスタム辞書による同義語・異表記の正規化

## インストール

### NPMパッケージとして使用（推奨）

```bash
npm install japanese-tfidf-embedder
```

### ソースからビルド

#### 必要な環境

- Rust 1.70+
- wasm-pack
- Node.js 14+ (オプション)
- Python 3 (デモサーバー用、オプション)

#### ビルド方法

```bash
# リポジトリのクローン
git clone https://github.com/satetsu888/japanese-tfidf-embedder.git
cd japanese-tfidf-embedder

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

### NPMパッケージの使用（ブラウザ/ESモジュール）

```javascript
import init, { IncrementalEmbedder, StableHashEmbedder } from 'japanese-tfidf-embedder';

async function initializeEmbedder() {
    // WASMモジュールの初期化（初回のみ必要）
    await init();
    
    // IncrementalEmbedderの作成
    const embedder = new IncrementalEmbedder(2.0);  // update_threshold=2.0（自動再学習を抑制）
    
    // 初期文書の追加
    const documents = [
        "機械学習は人工知能の一分野です",
        "深層学習はニューラルネットワークを使います",
        "自然言語処理で文書を解析します"
    ];
    
    for (const doc of documents) {
        embedder.add_document(doc, 64);  // embedding_dim=64
    }
    
    // モデルの訓練（重要：これをしないとベクトルが全て0になります）
    embedder.start_background_retrain(64);
    while (!embedder.step_retrain()) {
        // 訓練が完了するまで繰り返す
    }
    
    // 文書のベクトル化
    const vector = embedder.transform("AIの技術について");
    console.log("Vector dimensions:", vector.length);
    
    // 類似度計算
    const similarity = embedder.get_similarity(
        "機械学習とAI",
        "人工知能と深層学習"
    );
    console.log("Similarity:", similarity);
    
    return embedder;
}

// 使用例
initializeEmbedder().then(embedder => {
    // embedderを使った処理
    console.log("Documents:", embedder.get_unique_document_count());
});
```

### ユーザー辞書の使用

```javascript
import init, { IncrementalEmbedder } from 'japanese-tfidf-embedder';

async function setupWithDictionary() {
    await init();
    
    const embedder = new IncrementalEmbedder(2.0);
    
    // ユーザー辞書の定義
    const dictionary = [
        {
            surface: "人工知能",
            variants: ["AI", "エーアイ", "Artificial Intelligence"]
        },
        {
            surface: "機械学習",
            variants: ["ML", "マシンラーニング", "Machine Learning"]
        }
    ];
    
    // 辞書を適用
    embedder.set_dictionary(JSON.stringify(dictionary));
    
    // 文書を追加（異表記は自動的に正規化される）
    embedder.add_document("AIとMLの研究", 64);
    embedder.add_document("人工知能と機械学習の研究", 64);
    
    // モデルを訓練
    embedder.start_background_retrain(64);
    while (!embedder.step_retrain()) {}
    
    // "AI" と "人工知能" が同じトークンとして扱われるため、高い類似度になる
    const similarity = embedder.get_similarity(
        "AIの応用",
        "人工知能の応用"
    );
    console.log("Similarity with dictionary:", similarity);  // 高い値
    
    return embedder;
}
```

### Node.js環境での使用

```javascript
// CommonJSの場合
const init = require('japanese-tfidf-embedder');

(async () => {
    // 初期化
    const wasm = await init();
    const { IncrementalEmbedder } = wasm;
    
    const embedder = new IncrementalEmbedder(2.0);
    // 以下、ブラウザと同じように使用
})();
```

### HTMLでの直接使用

```html
<!DOCTYPE html>
<html>
<head>
    <script type="module">
        import init, { IncrementalEmbedder, StableHashEmbedder } from 'https://unpkg.com/japanese-tfidf-embedder/pkg/japanese_text_vector.js';
        
        async function run() {
            await init();
            
            // StableHashEmbedderの使用（文書追加に影響されない安定したベクトル）
            const stableEmbedder = new StableHashEmbedder(64, 2);
            const vector = stableEmbedder.transform("テキスト");
            console.log("Stable vector:", vector);
        }
        
        run();
    </script>
</head>
</html>
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
| `get_unique_document_count()` | ユニークな文書数を取得 |
| `contains_document(text)` | 文書が既に追加されているか確認 |
| `set_dictionary(json)` | ユーザー辞書を設定 |
| `clear_dictionary()` | ユーザー辞書をクリア |

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
| `set_dictionary(json)` | ユーザー辞書を設定 |
| `clear_dictionary()` | ユーザー辞書をクリア |

## 📊 パフォーマンス

| 指標 | 実測値 |
|------|--------|
| WASMサイズ | **184KB** |
| ベクトル化速度 | ~5ms/文書 |
| 語彙サイズ | 最大50,000語（動的調整） |
| 埋め込み次元 | 64-128（設定可能） |
| メモリ使用量 | 最小（wee_alloc使用） |
| SVD実装 | 高精度nalgebra SVD |

## 🔧 技術仕様

### 日本語トークナイザー

- **文字N-gram**: 
  - 2-gram, 3-gram の組み合わせ（一般的な文脈）
  - 漢字1-gram（個別漢字の意味、0.6倍の重み）
- **Stop-word フィルタリング**:
  - 助詞（は、が、を、に、で、と、の、へ等）
  - 助動詞（です、ます、だ、である等）
  - 形式名詞（こと、もの、ため等）
- **文字種別抽出**: ひらがな、カタカナ、漢字の連続抽出
- **単語境界推定**: 文字種変化による簡易的な単語分割
- **品質スコアリング**: 
  - 漢字熟語: 1.8倍
  - 単一漢字: 0.6倍
  - 混合文字種: 0.7倍
  - 助詞を含む: 0.5倍

### TF-IDF + LSA

- **TF-IDF**: 文書頻度による重み付け
- **LSA**: 高精度SVD（特異値分解）による次元削減
  - nalgebraの完全SVD実装
  - 特異値による重み付け
  - 主成分の正確な抽出
- **正規化**: L2正規化されたベクトル出力

## 📦 NPMパッケージの公開

### 公開手順

```bash
# 1. WASMビルド
npm run build

# 2. パッケージ公開（pkg内のpackage.jsonを使用）
cd pkg
npm publish

# または、ルートから直接実行
npm run publish  # build + publishを自動実行
```

### パッケージ構成

公開されるパッケージには以下のファイルが含まれます：
- `japanese_text_vector.js` - メインのJavaScriptラッパー
- `japanese_text_vector.d.ts` - TypeScript型定義
- `japanese_text_vector_bg.wasm` - WebAssemblyバイナリ
- `japanese_text_vector_bg.wasm.d.ts` - WASM型定義

## 🚀 GitHub Pages設定

このプロジェクトは、GitHub Actionsを使用して自動的にGitHub Pagesにデプロイされます。

### セットアップ手順

1. GitHubリポジトリの Settings → Pages へアクセス
2. Source を「GitHub Actions」に設定
3. mainブランチにpushすると自動的にデプロイされます

デプロイされたページ: https://satetsu888.github.io/japanese-tfidf-embedder/

## ライセンス

MIT License

## 謝辞

このプロジェクトは以下の技術を使用しています：

- [wasm-bindgen](https://github.com/rustwasm/wasm-bindgen)
- [nalgebra](https://nalgebra.org/)
- [serde](https://serde.rs/)
