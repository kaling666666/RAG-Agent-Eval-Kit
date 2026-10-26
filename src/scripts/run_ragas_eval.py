import os
import sys
import types

# ==========================================
# 🎯 补丁 1：修复找不到 src 的问题
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前在 src/scripts
project_root = os.path.dirname(os.path.dirname(current_dir))  # 往上两级到根目录

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ==========================================
# 🎯 补丁 2：修复 RAGAS 找不到 ChatVertexAI 的 Bug
# ==========================================
try:
    # 尝试从新包中导入
    from langchain_google_vertexai import ChatVertexAI

    # 伪造一个旧路径模块，欺骗 ragas 的导入语句
    fake_module = types.ModuleType('langchain_community.chat_models.vertexai')
    fake_module.ChatVertexAI = ChatVertexAI
    sys.modules['langchain_community.chat_models.vertexai'] = fake_module
    print("🔧 [Patch] 已成功应用 ChatVertexAI 兼容性补丁")
except ImportError:
    print("⚠️ [Warning] 未找到 langchain-google-vertexai 包，RAGAS 可能无法启动。")
    print(
        "请确保已在终端执行：D:\\environment\\miniconda3\\envs\\ragas-eval\\Scripts\\pip.exe install langchain-google-vertexai")

# ==========================================
# 正常导入 Adapter
# ==========================================
from src.adapters.ragas_adapter import RAGASAdapter


def main():
    print("🚀 [Script] 启动 RAGAS 自动化评测流水线...")

    # 1. 从环境变量读取配置
    base_url = os.getenv("EVAL_API_URL", "http://192.168.1.3:11434/v1")
    api_key = os.getenv("EVAL_API_KEY", "ollama")
    model_name = os.getenv("EVAL_MODEL", "qwen2.5:7b")
    embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    is_local = os.getenv("IS_LOCAL", "True").lower() in ("true", "1", "t")

    # 2. 定义数据路径 (注意这里匹配你的真实目录结构)
    dataset_path = os.path.join(project_root, "datasets", "rag_test_data", "rag_test_data.json")
    output_path = os.path.join(project_root, "outputs", "ragas_results.json")

    if not os.path.exists(dataset_path):
        print(f"❌ [Script] 致命错误：找不到测试数据集 {dataset_path}")
        print("💡 请确保 datasets/rag_test_data/rag_test_data.json 存在。")
        sys.exit(1)

    # 3. 实例化适配器并执行评测
    try:
        adapter = RAGASAdapter(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            embedding_model_name=embedding_model,
            is_local=is_local
        )

        adapter.run_evaluation(
            dataset_path=dataset_path,
            output_path=output_path
        )
        print("🎉 [Script] RAGAS 评测流水线执行完毕！")

    except Exception as e:
        print(f"\n❌ [Script] 评测流程中断，错误信息: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()