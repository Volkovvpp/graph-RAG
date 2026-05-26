from pathlib import Path


def get_prompt(prompt_file_name: str) -> str:
    """
    Load a prompt template from the prompts directory.

    Args:
        prompt_file_name: The name of the file (e.g., 'extraction.txt' or 'extraction').

    Returns:
        The content of the prompt file as a string.
    """
    prompts_dir = Path(__file__).parent / "prompts"
    prompt_path = prompts_dir / prompt_file_name

    # Try adding .txt if not present
    if not prompt_path.exists() and not prompt_path.suffix:
        prompt_path = prompt_path.with_suffix(".txt")

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    return prompt_path.read_text(encoding="utf-8")
