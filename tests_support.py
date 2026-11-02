"""Small deterministic inference substitutes; never used by the application CLI."""

from transformers import PreTrainedTokenizerBase


class FakeTokenizer(PreTrainedTokenizerBase):
    """Count characters so boundary tests do not require external model files."""

    # Represent text and an optional passage as one synthetic token per character, allowing
    # token-budget tests to run without downloading model tokenizers.
    def encode(self, text: str, passage: str = "", **kwargs) -> list:
        """Return one synthetic token per character."""

        return list(text + passage)

    # Expose character tokens through the interface expected by the Hugging Face text
    # splitter so lifecycle tests exercise actual splitting with predictable lengths.
    def tokenize(self, text: str, **kwargs) -> list:
        """Support the splitter's Hugging Face token-count interface."""

        return list(text)


class FakeEmbedder:
    """Provide fixed vectors for lifecycle and access tests."""

    dimension = 3
    max_tokens = 256
    tokenizer = FakeTokenizer()

    # Accept the production embedder configuration signature without loading weights; fixed
    # class attributes provide the tokenizer, dimension and capacity used by tests.
    def __init__(self, config) -> None:
        """Accept the same configuration as the real embedder."""

    # Count characters as synthetic tokens so deterministic tests can check chunk sizes
    # independently of external model files.
    def token_count(self, text: str) -> int:
        """Count synthetic tokens."""

        return len(text)

    # Return the same three-dimensional unit vector for each input, ensuring persistence and
    # access-boundary tests cannot pass through semantic vector separation.
    def encode(self, texts: list[str]) -> list[list[float]]:
        """Return deterministic unit vectors."""

        return [[1.0, 0.0, 0.0] for text in texts]


class FakeReranker:
    """Keep input ordering for lifecycle tests."""

    max_seq_length = 512
    tokenizer = FakeTokenizer()

    # Accept production model arguments without constructing a real model, letting retrieval
    # lifecycle tests substitute this deterministic scorer.
    def __init__(self, *args, **kwargs) -> None:
        """Accept the real reranker's model options."""

    # Assign descending scores in candidate order so tests retain reproducible ranking while
    # exercising the production reranking and selection flow.
    def predict(self, pairs: list, batch_size: int) -> list[float]:
        """Assign deterministic descending scores."""

        return [float(len(pairs) - position) for position in range(len(pairs))]
