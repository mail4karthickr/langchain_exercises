from pathlib import Path

class FileUtils:
    @staticmethod
    def contents(file_name: str, base_dir: str = ".") -> str:
        """
        Reads the contents of a .txt file from a specified base directory.

        Args:
            file_name (str): Name of the file (without extension).
            base_dir (str): Relative or absolute directory path to look in. Default is current dir.

        Returns:
            str: File contents.

        Raises:
            FileNotFoundError: If the file is not found in the given directory.
        """
        file_path = Path(base_dir).resolve() / f"{file_name}.txt"
        if not file_path.exists():
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")
        return file_path.read_text(encoding="utf-8").strip()