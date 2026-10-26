import os
import json
import warnings
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 忽略 RAGAS 内部的一些无关警告信息，保持日志整洁
warnings.filterwarnings("ignore")


class RAGASAdapter:
    """
    RAGAS 评测适配器
    负责将统一的 JSON 数据转换为 RAGAS 所需的 Dataset 格式，并执行四大核心指标评测。
    """

    def __init__(self, base_url: str, api_key: str, model_name: str, embedding_model_name: str = "text-embedding-3-small", is_local: bool = False):
        """
        初始化 RAGAS 适配器
        :param base_url: 裁判模型 API 地址 (如 http://host.docker.internal:11434/v1)
        :param api_key: 鉴权 Token
        :param model_name: 生成模型名称 (如 qwen2.5:7b 或 deepseek-chat)
        :param embedding_model_name: 向量模型名称
        :param is_local: 是否使用本地 Ollama 的 Embedding
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name

        # 强制将配置注入环境变量，防止 RAGAS 默认去连 OpenAI
        os.environ["OPENAI_API_KEY"] = self.api_key
        os.environ["OPENAI_API_BASE"] = self.base_url

        print(f"🔧 [RAGAS Adapter] 初始化裁判模型: {self.model_name} (API: {self.base_url})")

        # 配置 LLM (裁判模型)
        self.judge_llm = ChatOpenAI(
            model=self.model_name,
            temperature=0,
            api_key=self.api_key,
            base_url=self.base_url
        )

        # 配置 Embeddings (RAGAS 计算 answer_relevancy 等指标必须用到向量化)
        if is_local:
            # 如果使用本地 Ollama
            try:
                from langchain_community.embeddings import OllamaEmbeddings
                self.judge_embeddings = OllamaEmbeddings(
                    model=embedding_model_name,
                    base_url=self.base_url.replace("/v1", "")  # Ollama 本地接口不需要 /v1
                )
                print(f"🔧 [RAGAS Adapter] 使用本地向量模型: {embedding_model_name}")
            except ImportError:
                raise ImportError("使用本地 Embedding 需要安装 langchain-community 库，请执行 pip install langchain-community")
        else:
            # 兼容云端 API (如 OpenAI 等)
            self.judge_embeddings = OpenAIEmbeddings(
                model=embedding_model_name,
                api_key=self.api_key,
                base_url=self.base_url
            )
            print(f"🔧 [RAGAS Adapter] 使用云端向量模型: {embedding_model_name}")

    def run_evaluation(self, dataset_path: str, output_path: str):
        """
        执行评测的主流程
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"❌ 找不到测试数据集: {dataset_path}")

        print(f"📂 [RAGAS Adapter] 正在加载数据: {dataset_path}")
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 转换为 RAGAS 需要的 HuggingFace Dataset 格式
        try:
            dataset = Dataset.from_list(data)
        except Exception as e:
            raise ValueError(f"❌ 数据格式错误，请确保 JSON 是列表字典格式: {e}")

        print(f"✅ [RAGAS Adapter] 成功加载 {len(dataset)} 条数据，开始评测...")

        try:
            # 执行 RAGAS 评测
            result = evaluate(
                dataset=dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall
                ],
                llm=self.judge_llm,
                embeddings=self.judge_embeddings
            )

            print("\n" + "=" * 40)
            print("📊 RAGAS 评测结果")
            print("=" * 40)
            print(result)

            # 处理并保存结果
            self._save_results(result, output_path)
            return result

        except Exception as e:
            print(f"❌ [RAGAS Adapter] 评测执行失败: {e}")
            raise

    def _save_results(self, result, output_path: str):
        """将评测结果保存为 JSON"""
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 将 RAGAS 的 Result 对象转为 Pandas DataFrame 再转为字典
        result_df = result.to_pandas()
        records = result_df.to_dict(orient="records")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=4)

        print(f"📁 [RAGAS Adapter] 详细评测记录已保存至: {output_path}")


if __name__ == "__main__":
    # --- 本地测试代码 (方便你在本地环境单独调试该模块) ---
    # 模拟从环境变量获取配置
    BASE_URL = os.getenv("EVAL_API_URL", "http://192.168.1.3:11434/v1")
    API_KEY = os.getenv("OPENAI_API_KEY", "ollama")
    MODEL = os.getenv("EVAL_MODEL", "qwen2.5:7b")

    # 当前项目根目录推导
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) # src/adapters
    # 🎯 因为当前文件在 src/adapters/ 下，往上找3级才是项目根目录 (adapters -> src -> root)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

    # 匹配你真实的目录结构：datasets/rag_test_data/rag_test_data.json
    DATASET_FILE = os.path.join(PROJECT_ROOT, "datasets", "rag_test_data", "rag_test_data.json")
    OUTPUT_FILE = os.path.join(PROJECT_ROOT, "outputs", "ragas_results.json")

    # 如果你在本地用 Ollama，请将 is_local 设为 True
    adapter = RAGASAdapter(
        base_url=BASE_URL,
        api_key=API_KEY,
        model_name=MODEL,
        embedding_model_name="nomic-embed-text",  # 如果你用本地 Ollama，需要这个向量模型
        is_local=True
    )

    adapter.run_evaluation(dataset_path=DATASET_FILE, output_path=OUTPUT_FILE)