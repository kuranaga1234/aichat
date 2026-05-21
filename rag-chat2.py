import os
import argparse
import re
import unicodedata
import hashlib
from pathlib import Path
from llama_cpp import Llama

# === RAG ===
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# ==================================================
# 定数
# ==================================================
VECTORDB_PATH = "./vectordb_new"
MODEL_DIR = "./models"
SUPPORTED_EXT = [".txt", ".md", ".json", ".jsonl"]

SIM_HIGH = 0.8
SIM_MID = 0.5

DIV_TOKEN_SIZE = 400
DIV_TOKEN_OVERLAP = 50
CONFRICT_MAX_COUNT = 1000

REJECT_WORDS = ["お答えできません", "申し訳ありません", "回答できません"]

# ==================================================
# 共通ユーティリティ
# ==================================================
def normalize_text(text: str) -> str:
    """テキストの正規化 """
    # NFKCで半角カナを全角カナへ、かつ全角英数を半角英数へ一旦統一
    text = unicodedata.normalize("NFKC", text)
    
    # 半角文字（英数字・記号 0x21-0x7E）を全角（0xFF01-0xFF5E）に変換
    # 半角スペース（0x20）も全角スペース（\u3000）に変換
    text = "".join(
        chr(ord(c) + 0xfee0) if 0x21 <= ord(c) <= 0x7E else
        "\u3000" if c == " " else c
        for c in text
    )
    
    # カタカナをひらがなに変換
    text = "".join(
        chr(ord(c) - 0x60) if "\u30a1" <= c <= "\u30f6" else c for c in text
    )

    # 英字をすべて小文字に変換
    text = text.lower()

    # 連続する全角空白を1つにまとめる
    text = re.sub(r"[\s\u3000]+", "\u3000", text)
    return text.strip()


def hash6(text: str) -> int:
    """テキストから6桁の整数ハッシュを生成 """
    h = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)
    return h % 1000000


def recursive_split(text: str, size=DIV_TOKEN_SIZE, overlap=DIV_TOKEN_OVERLAP):
    """再帰的分割 """
    if len(text) <= size:
        return [text]

    chunks, buf = [], ""
    for sent in re.split(r"([。！？\n])", text):
        if len(buf) + len(sent) <= size:
            buf += sent
        else:
            if buf:
                chunks.append(buf)
            buf = buf[-overlap:] + sent if overlap < len(buf) else sent
    if buf:
        chunks.append(buf)
    return [c for c in chunks if c.strip()]


# ==================================================
# VectorDB クラス
# ==================================================
class VectorDB:
    def __init__(self, modelv: str):
        if not RAG_AVAILABLE:
            raise RuntimeError("ChromaDB または SentenceTransformer がインストールされていません。")

        # モデルIDの決定 
        self.model_prefix = "000001" if modelv == "small" else "000002"
        model_name = "intfloat/multilingual-e5-small" if modelv == "small" else "intfloat/multilingual-e5-base"
        
        print(f"Loading Embedding Model: {model_name}...")
        self.embedder = SentenceTransformer(model_name)
        
        self.client = chromadb.PersistentClient(path=VECTORDB_PATH)
        self.category_col = self.client.get_or_create_collection("category")
        self.search_col = self.client.get_or_create_collection("search")
        self.errors = 0

    def clear_all(self):
        """全クリア """
        try:
            self.client.delete_collection("category")
            self.client.delete_collection("search")
        except:
            pass
        self.category_col = self.client.get_or_create_collection("category")
        self.search_col = self.client.get_or_create_collection("search")

    def _generate_unique_id(self, base_hash: int) -> str:
        """ID衝突回避ロジック（最大1000回インクリメント） """
        for i in range(CONFRICT_MAX_COUNT):
            target_id = (base_hash + i) % 1000000
            str_id = str(target_id).zfill(6)
            full_id = self.model_prefix + str_id
            
            # カテゴリテーブルに存在するかチェック
            exists = self.category_col.get(ids=[full_id])
            if not exists or not exists["ids"]:
                return full_id
        
        raise RuntimeError("ID衝突が1000回を超えました。登録を停止します。 ")

    def register(self, llm, path: str, update=False):
        """登録処理のメインルーチン """
        p = Path(path)
        if not p.exists():
            print(f"Error: {path} が見つかりません。")
            return

        files = [p] if p.is_file() else [
            f for f in p.iterdir() if f.suffix in SUPPORTED_EXT
        ]
        
        for f in files:
            try:
                self._register_file(llm, f, update)
            except Exception as e:
                self.errors += 1
                print(f"ファイル {f.name} の登録中にエラー: {e}")
                if self.errors >= 5:
                    raise RuntimeError("エラー発生回数が制限を超えました。 ")

    def _register_file(self, llm, file: Path, update: bool):
        raw = file.read_text(encoding="utf-8", errors="replace")
        raw = raw.replace("\r\n", "\n").replace("\r", "\n")
        norm = normalize_text(raw)

        # 1. カテゴリ要約の取得 
        prompt = f"次の内容からカテゴリを表す要約を50文字以内で作成してください。\n{raw[:500]}"
        res = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        summary = res["choices"][0]["message"]["content"].strip()[:50]

        # 2. 既存カテゴリのセマンティック検索 
        vec_sum = self.embedder.encode([f"passage: {summary}"]).tolist()
        dup = self.category_col.query(vec_sum, n_results=1)

        # 類似度判定 (距離が小さいほど類似)
        if dup["distances"] and dup["distances"][0] and dup["distances"][0][0] < (1 - SIM_HIGH):
            final_id = dup["ids"][0][0]
            print(f"既存カテゴリを採用: {final_id} ({summary})")
        else:
            # 新規ID発行（衝突回避付き） 
            base_h = hash6(summary)
            final_id = self._generate_unique_id(base_h)
            self.category_col.add(
                ids=[final_id],
                embeddings=vec_sum,
                documents=[summary],
                metadatas=[{"cid": final_id[-6:]}]
            )
            print(f"新規カテゴリ登録: {final_id} ({summary})")

        # 3. 検索テーブルへの登録 
        if update:
            old = self.search_col.get(where={"url": str(file)})
            if old["ids"]:
                self.search_col.delete(old["ids"])

        chunks = recursive_split(norm)
        for i, c in enumerate(chunks):
            vec = self.embedder.encode([f"query: {c}"]).tolist()
            self.search_col.add(
                ids=[f"{final_id}_{i}"],
                embeddings=vec,
                documents=[c],
                metadatas=[{"cid": final_id[-6:], "url": str(file)}]
            )

    def search(self, query: str):
        """セマンティック検索 """
        q = normalize_text(query)
        vec = self.embedder.encode([f"query: {q}"]).tolist()
        # 類似度上位3件取得
        res = self.search_col.query(vec, n_results=3)
        return res


# ==================================================
# メイン処理
# ==================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regdb", help="新規登録（全クリア）")
    ap.add_argument("--upddb", help="追記・更新")
    ap.add_argument("--file", help="システムプロンプト用ファイル ")
    ap.add_argument("--rag", type=int, default=1, help="RAGモード(0,1,2,3)")
    ap.add_argument("--model", default="gemma4", help="応答LLM")
    ap.add_argument("--modelv", default="small", choices=["small", "base"], help="Embeddingモデル")
    args = ap.parse_args()

    llm = None

    # --- DB登録モード ---
    if args.regdb or args.upddb:
        print("Initializing LLM for Summarization...")
        llm = Llama.from_pretrained(
            repo_id="Bartowski/gemma-2-2b-it-GGUF",
            filename="*Q4_K_M.gguf",
            local_dir=MODEL_DIR,
            n_ctx=4096
        )
        vdb = VectorDB(args.modelv)
        if args.regdb:
            vdb.clear_all()
            vdb.register(llm, args.regdb, update=False)
        else:
            vdb.register(llm, args.upddb, update=True)
        print("DB処理が完了しました。")

    # --- チャットモード ---
    if llm is None:
        print(f"Loading Chat LLM ({args.model})...")
        # 設計書のモデル指定に合わせてロード（ここではgemma2を例に）
        llm = Llama.from_pretrained(
            repo_id="Bartowski/gemma-2-2b-it-GGUF",
            filename="*Q4_K_M.gguf",
            local_dir=MODEL_DIR,
            n_ctx=4096
        )
    else:
        print("Using the already loaded LLM for Chat...")

    # システムプロンプトの読み込み 
    system_instruction = "あなたは優秀なアシスタントです。"
    if args.file:
        f_path = Path(args.file)
        if f_path.exists():
            system_instruction = f_path.read_text(encoding="utf-8")
            print(f"システムプロンプトをファイルから読み込みました: {args.file}")

    if args.rag != 0:
        system_instruction += "\n関連情報がない場合は『関連情報が見つかりませんでした』と答えてください。"

    vdb = VectorDB(args.modelv) if args.rag != 0 else None
    history = [{"role": "system", "content": system_instruction}]

    print("--- チャットを開始します (exitで終了) ---")
    while True:
        user_input = input("\nあなた> ").strip()
        if user_input.lower() in ("exit", "quit"):
            break

        context = ""
        raw_docs = ""
        
        # RAG検索処理 
        if vdb and args.rag in (1, 2):
            res = vdb.search(user_input)
            if res["distances"] and res["distances"][0]:
                best_dist = res["distances"][0][0]
                docs = res["documents"][0]
                
                # 距離を類似度に簡易変換 (ChromaDBのデフォルトはL2距離)
                # 0.8以上（距離0.2以下）
                if best_dist <= (1 - SIM_HIGH):
                    raw_docs = "\n".join(docs)
                    context = f"【関連情報】\n{raw_docs}\n"
                # 0.5〜0.8（距離0.2〜0.5）
                elif best_dist <= (1 - SIM_MID):
                    raw_docs = "\n".join(docs)
                    print(f"!! 確信度が低いため確認中... (参考情報あり)")
                    context = f"【参考情報（要確認）】\n{raw_docs}\n"

        # プロンプト組み立て
        final_prompt = f"{context}質問: {user_input}"
        history.append({"role": "user", "content": final_prompt})

        # LLM推論
        response = llm.create_chat_completion(messages=history, max_tokens=1024)
        answer = response["choices"][0]["message"]["content"]

        # 拒否回答検知とリカバリ 
        if any(word in answer for word in REJECT_WORDS) and raw_docs:
            answer = "LLMによる要約に失敗したため、参考ドキュメントをそのまま表示します。\n" + raw_docs

        print(f"\nAI> {answer}")
        history.append({"role": "assistant", "content": answer})

        # 履歴が長すぎる場合の簡易カット（直近10件）
        if len(history) > 11:
            history = [history[0]] + history[-10:]

if __name__ == "__main__":
    main()