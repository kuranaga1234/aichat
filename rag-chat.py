import os
import argparse
import json
import re
import unicodedata
from llama_cpp import Llama

# RAG関連のインポート
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import logging
    logging.set_verbosity_error()
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# --- 設定定数 ---
USE_LOCAL_MODEL = True
MODEL_DIR = "./models"
VECTORDB_PATH = "./my_vectordb"
DATASET_JSONL = "./dataset.jsonl"

# RAG用の定数
DIST_NO_MATCH = 0.5
DIST_CANDIDATE = 0.25
RAG_CATEGORIES = ["人事", "通勤", "給与", "その他"]
CATEGORY_KEYWORDS = {
    "人事": [
        "人事", "評価", "等級", "降格", "昇格", "昇進", "制度", "職種", "スペシャリスト",
        "マネジメント", "ランク", "資格手当", "新卒", "コアバリュー", "市場価値",
        "スキル", "採用", "入社", "退職", "規定", "問い合わせ", "フィードバック",
    ],
    "通勤": [
        "通勤", "交通", "定期", "定期券", "バス", "電車", "鉄道", "バス停", "経路",
        "らくらく", "らくらくBOSS", "申請", "承認", "代理", "手当", "実費",
        "徒歩", "紛失", "割引", "障害者", "2km", "2Km",
    ],
    "給与": [
        "給与", "給料", "賞与", "ボーナス", "基本給", "昇給", "減給", "手当",
        "業績手当", "所得税", "非課税", "単価", "チャージ", "月給", "季節賞与",
        "決算賞与", "支給", "報酬",
    ],
    "その他": [
        "営業時間", "営業", "支払い", "支払い方法", "返品", "商品", "到着",
        "クレジットカード", "銀行振込", "電子マネー", "未開封", "キャンセル",
    ],
}

MODELS = {
    # 大型モデル
    "gemma": {"name": "Gemma-2-9B", "repo_id": "Bartowski/gemma-2-9b-it-GGUF", "filename": "*Q4_K_M.gguf", "n_ctx": 4096, "type": "llama_cpp", "enable_rag": True},
    "gemma4-text": {"name": "Gemma-4-E4B-GGUF", "filename": "google_gemma-4-E4B-it-Q4_K_M.gguf", "n_ctx": 4096, "type": "llama_cpp", "enable_rag": False},
    "gemma4-rag": {"name": "Gemma-4-E4B-GGUF（RAG有効）", "filename": "google_gemma-4-E4B-it-Q4_K_M.gguf", "n_ctx": 4096, "type": "llama_cpp", "enable_rag": True},
    "elyza": {"name": "Llama-3-ELYZA-JP-8B", "repo_id": "elyza/Llama-3-ELYZA-JP-8B-GGUF", "filename": "Llama-3-ELYZA-JP-8B-q4_k_m.gguf", "n_ctx": 4096, "type": "llama_cpp", "enable_rag": True},
    # 軽量モデル
    "llama3b": {"name": "Llama-3.2-3B", "repo_id": "Bartowski/Llama-3.2-3B-Instruct-GGUF", "filename": "*Q4_K_M.gguf", "n_ctx": 16384, "type": "llama_cpp", "enable_rag": True}
}

# ─────────────────────────────────────────────
# RAG用ユーティリティ関数
# ─────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """テキストを正規化して表記ゆれを吸収"""
    text = unicodedata.normalize("NFKC", text)
    # カタカナ→ひらがな
    text = "".join(
        chr(ord(ch) - 0x60) if "\u30a1" <= ch <= "\u30f6" else ch
        for ch in text
    )
    text = text.lower()
    text = re.sub(r"[！!。、〜～ー\-・\s]", "", text)
    return text


def detect_categories(query: str) -> list[str]:
    """クエリから検索対象カテゴリを判定"""
    matched = []
    for category, keywords in CATEGORY_KEYWORDS.items():
        norm_query = normalize_text(query)
        if any(normalize_text(kw) in norm_query for kw in keywords):
            matched.append(category)
    
    if not matched:
        return RAG_CATEGORIES
    return matched


def collection_name(category: str) -> str:
    """カテゴリ名をコレクション名に変換"""
    mapping = {
        "人事": "qa_jinji",
        "通勤": "qa_tsukin",
        "給与": "qa_kyuyo",
        "その他": "qa_sonota",
    }
    return mapping.get(category, "qa_sonota")


# ─────────────────────────────────────────────
# VectorDB クラス
# ─────────────────────────────────────────────

class VectorDB:
    """ChromaDB を使用した質問回答ベクトルデータベース"""
    
    def __init__(self, db_path=VECTORDB_PATH):
        if not RAG_AVAILABLE:
            print("警告: chromadb または sentence_transformers がインストールされていません")
            self.available = False
            return
        
        self.model = SentenceTransformer('intfloat/multilingual-e5-small')
        self.client = chromadb.PersistentClient(path=db_path)
        self.collections = {}
        
        for category in RAG_CATEGORIES:
            self.collections[category] = self.client.get_or_create_collection(
                name=collection_name(category)
            )
        self.available = True

    def register_from_jsonl(self, jsonl_path):
        """JSONLファイルからデータを登録"""
        if not self.available:
            print("エラー: VectorDBが利用不可です")
            return False
        
        if not os.path.exists(jsonl_path):
            print(f"エラー: {jsonl_path} が見つかりません")
            return False

        # 既存データをクリア
        for category in RAG_CATEGORIES:
            try:
                # 既存コレクションを削除して再作成
                self.client.delete_collection(name=collection_name(category))
                self.collections[category] = self.client.get_or_create_collection(
                    name=collection_name(category)
                )
            except:
                pass

        # カテゴリ別にデータを仕分け
        data_by_category = {
            cat: {"questions": [], "documents": [], "metadatas": [], "ids": []}
            for cat in RAG_CATEGORIES
        }

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            global_id = 0
            for line in f:
                try:
                    data = json.loads(line)
                    question = data.get('question')
                    answer = data.get('answer')
                    category = data.get('category', 'その他')
                    if not question or not answer:
                        continue
                    if category not in RAG_CATEGORIES:
                        category = 'その他'
                except json.JSONDecodeError:
                    continue

                bucket = data_by_category[category]
                bucket["questions"].append(f"passage: {question}")
                bucket["documents"].append(question)
                original_question = data.get('original_question', question)
                bucket["metadatas"].append({
                    "answer": answer,
                    "category": category,
                    "original_question": original_question
                })
                bucket["ids"].append(f"id_{global_id}")
                global_id += 1

        # カテゴリごとにエンコード＆登録
        for category, bucket in data_by_category.items():
            if not bucket["questions"]:
                continue
            embeddings = self.model.encode(
                bucket["questions"],
                batch_size=32,
                show_progress_bar=True,
            ).tolist()
            self.collections[category].add(
                documents=bucket["documents"],
                embeddings=embeddings,
                metadatas=bucket["metadatas"],
                ids=bucket["ids"],
            )
            print(f"  [{category}] {len(bucket['documents'])}件 登録完了")

        print("\n✓ すべてのカテゴリの登録が完了しました")
        return True

    def search(self, query: str, n_results=3) -> list[dict]:
        """クエリから関連する情報を検索"""
        if not self.available:
            return []

        # 検索対象カテゴリを判定
        target_categories = detect_categories(query)
        
        # E5モデルのルール: 質問側には "query: " をつける
        query_vector = self.model.encode([f"query: {query}"]).tolist()

        # 結果を収集
        all_results = []
        for category in target_categories:
            col = self.collections[category]
            if col.count() == 0:
                continue
            results = col.query(
                query_embeddings=query_vector,
                n_results=min(n_results, col.count()),
            )
            if not results['ids'][0]:
                continue
            for question, meta, distance in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0],
            ):
                all_results.append({
                    "distance": distance,
                    "question": meta.get('original_question', question),
                    "answer": meta.get('answer', ''),
                    "category": meta.get('category', category)
                })

        # 距離でソート
        all_results.sort(key=lambda x: x['distance'])
        return all_results

    def format_context(self, query: str) -> str:
        """RAGから得られたコンテキストをフォーマット"""
        if not self.available:
            return ""
        
        results = self.search(query)
        if not results:
            return ""

        closest = results[0]['distance'] if results else 1.0
        
        # 距離が0.5より大きい → 関連情報なし
        if closest > DIST_NO_MATCH:
            return ""
        
        # 距離が0.25より大きい → 複数候補を表示
        elif closest > DIST_CANDIDATE:
            context = "[関連する情報]（候補）:\n"
            for i, result in enumerate(results[:3], 1):
                context += f"{i}. 【{result['category']}】 質問: {result['question']}\n   回答: {result['answer']}\n\n"
            return context
        
        # 距離が0.25以下 → 最良の回答を返す
        else:
            context = "[関連する情報]:\n"
            context += f"【{results[0]['category']}】 Q: {results[0]['question']}\n"
            context += f"A: {results[0]['answer']}\n\n"
            return context

# ─────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Llama-cpp AI Chat with RAG Integration")
    parser.add_argument("--model", type=str, default="llama3b", choices=list(MODELS.keys()))
    parser.add_argument("--file", type=str, help="読み込むファイルのパスを指定してください")
    parser.add_argument("--regdb", action="store_true", help="ChromaDBへのデータ登録モード")
    parser.add_argument("--dataset", type=str, default=DATASET_JSONL, help="JSONLデータセットのパス")
    args = parser.parse_args()

    # --- データベース登録モード ---
    if args.regdb:
        print("=== ChromaDB データ登録モード ===")
        if not RAG_AVAILABLE:
            print("エラー: chromadb と sentence_transformers をインストールしてください")
            print("実行: pip install chromadb sentence-transformers")
            return
        
        if not os.path.exists(args.dataset):
            print(f"エラー: {args.dataset} が見つかりません")
            return
        
        print(f"読み込み: {args.dataset}")
        vdb = VectorDB(VECTORDB_PATH)
        success = vdb.register_from_jsonl(args.dataset)
        if success:
            print("✓ データベース登録完了")
        return

    # --- 通常のチャットモード ---
    print("=== Llama-cpp AI チャットモード ===")

    # RAG初期化（モデル設定で有効/無効を判定）
    vdb = None
    selected = MODELS[args.model]
    enable_rag_for_model = selected.get("enable_rag", True)
    
    if enable_rag_for_model and RAG_AVAILABLE:
        vdb = VectorDB(VECTORDB_PATH)
        if not any(vdb.collections[cat].count() > 0 for cat in RAG_CATEGORIES):
            print("⚠ RAGデータベースが空です")
            print(f"  実行: python {__file__} --regdb")
            vdb = None
        else:
            counts = {cat: vdb.collections[cat].count() for cat in RAG_CATEGORIES}
            print(f"✓ RAG有効 ({sum(counts.values())}件)")
    else:
        if enable_rag_for_model:
            print("⚠ RAG無効（chromadbをインストールしてください）")
        else:
            print(f"ℹ RAG無効（--{args.model}では無効です）")

    # ファイル読み込み
    file_content = ""
    if args.file:
        if os.path.exists(args.file):
            try:
                with open(args.file, "r", encoding="utf-8") as f:
                    file_content = f.read()
                print(f"✓ ファイル '{args.file}' を読み込みました")
            except Exception as e:
                print(f"ファイル読み込みエラー: {e}")
        else:
            print(f"警告: ファイル '{args.file}' が見つかりません")

    # テキストモデルロード
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    print(f"\n--- ロード中: {selected['name']} (n_ctx: {selected['n_ctx']}) ---")
    
    llm = None
    
    try:
        n_batch = 512
        
        if "filename" not in selected:
            print(f"エラー: モデル設定が不正です")
            return
        
        if "repo_id" in selected:
            # HuggingFaceから取得
            llm = Llama.from_pretrained(
                repo_id=selected["repo_id"],
                filename=selected["filename"],
                local_dir=MODEL_DIR,
                offline=USE_LOCAL_MODEL,
                verbose=False,
                n_ctx=selected["n_ctx"],
                n_batch=n_batch
            )
        else:
            # ローカルファイルから直接読み込み
            model_path = os.path.join(MODEL_DIR, selected["filename"])
            llm = Llama(
                model_path=model_path,
                verbose=False,
                n_ctx=selected["n_ctx"],
                n_batch=n_batch
            )
    except Exception as e:
        print(f"エラー: {e}")
        return

    history = []
    print(f"\n--- {selected['name']} とのチャット開始 ---")
    print(f"  RAG検索: {'有効' if vdb else '無効'}")
    print(f"  ファイルコンテキスト: {'あり' if file_content else 'なし'}")

    while True:
        user_input = input("\nあなた: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ['exit', 'quit', 'bye', '終了', 'えぃｔ', 'くいｔ', 'びぇ', 'しゅうりょう']:
            break

        # ===== プロンプト構築 =====
        prompt = user_input
        
        if len(history) == 0:
            # 初回のみシステムプロンプトを設定
            instruction = "あなたは優秀なアシスタントです。"

            # RAGコンテキストを追加
            rag_context = ""
            if vdb:
                rag_context = vdb.format_context(user_input)
            
            # ファイルコンテキストを追加
            file_context = ""
            if file_content:
                file_context = f"\n[提供されたファイルの内容]:\n{file_content}\n"
            
            # 総合プロンプト
            if vdb:
                instruction += " ユーザーからの質問に対して、関連する情報があれば提供してください。関連する情報がない場合は「関連する情報が見つかりませんでした」と応答してください。"
            prompt = (
                f"{instruction}\n"
                f"{rag_context}"
                f"{file_context}"
                f"\n[ユーザーの質問]: {user_input}"
            )
            print(f"\n--- プロンプト ---\n{prompt}\n--- ---")

        history.append({"role": "user", "content": prompt})

        try:
            response = llm.create_chat_completion(messages=history, max_tokens=512)
            ai_answer = response["choices"][0]["message"]["content"]
            
            print(f"\nAI ({selected['name']}): {ai_answer}")
            history.append({"role": "assistant", "content": ai_answer})
        except Exception as e:
            print(f"\nエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            break


if __name__ == "__main__":
    main()
