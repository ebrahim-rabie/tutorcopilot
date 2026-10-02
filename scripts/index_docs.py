import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.embedding import get_embedder
from app.index import Index
from app.settings import Settings


def main():
    settings = Settings()
    embedder = get_embedder(settings)
    index = Index(settings, embedder)
    index.build_from_directory()
    index.save()
    print(f"Indexed {len(index.chunks)} chunks -> {settings.index_dir}")
    print("Sources:", sorted({c.source for c in index.chunks}))


if __name__ == "__main__":
    main()
