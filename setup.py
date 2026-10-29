"""
setup.py
========
Script de configuração do Solar Weather Assistant.
Verifica dependências, cria estrutura de pastas e indexa os documentos.

Uso:
    python setup.py
"""

import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def check_python():
    if sys.version_info < (3, 9):
        print("❌ Python 3.9+ é necessário.")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")


def check_env():
    key = os.environ.get("GROQ_API_KEY", "")
    if not key or not key.startswith("gsk_"):
        print("⚠️  GROQ_API_KEY não encontrada ou inválida.")
        print("   Defina com: export GROQ_API_KEY='gsk_...'")
        print("   Obtenha em: https://console.groq.com/keys")
        return False
    print("✅ GROQ_API_KEY configurada.")
    return True


def install_deps():
    print("📦 Instalando dependências...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements.txt")],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"❌ Erro ao instalar dependências:\n{result.stderr}")
        sys.exit(1)
    print("✅ Dependências instaladas.")


def check_docs():
    docs_dir = ROOT / "docs"
    docs_dir.mkdir(exist_ok=True)
    files = list(docs_dir.glob("*.md")) + list(docs_dir.glob("*.pdf")) + list(docs_dir.glob("*.txt"))
    if not files:
        print("⚠️  Pasta docs/ está vazia. Adicione documentos antes de indexar.")
        return False
    print(f"✅ {len(files)} documentos em docs/:")
    for f in sorted(files):
        print(f"   - {f.name}")
    return True


def run_ingest():
    print("\n🔍 Indexando documentos no ChromaDB...")
    result = subprocess.run(
        [sys.executable, str(ROOT / "src" / "ingest.py")],
        capture_output=False
    )
    if result.returncode != 0:
        print("❌ Erro na indexação. Verifique o erro acima.")
        sys.exit(1)
    print("✅ Indexação concluída!")


def main():
    print("=" * 50)
    print("  Solar Weather Assistant — Setup")
    print("=" * 50)

    check_python()
    has_key = check_env()
    install_deps()
    has_docs = check_docs()

    if not has_docs:
        print("\n❌ Adicione documentos à pasta docs/ e rode novamente.")
        sys.exit(1)

    chroma_dir = ROOT / ".chroma"
    if chroma_dir.exists() and any(chroma_dir.iterdir()):
        print(f"\n⚠️  .chroma/ já existe ({sum(1 for _ in chroma_dir.rglob('*'))} arquivos).")
        resp = input("   Reindexar? [s/N]: ").strip().lower()
        if resp != "s":
            print("   Pulando indexação.")
        else:
            import shutil
            shutil.rmtree(chroma_dir)
            run_ingest()
    else:
        run_ingest()

    if not has_key:
        print("\n⚠️  Configure GROQ_API_KEY e rode: python app.py")
    else:
        print("\n🚀 Tudo pronto! Inicie o app com:")
        print("   python app.py")
    print("=" * 50)


if __name__ == "__main__":
    main()
