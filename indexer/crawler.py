# indexer/crawler.py

import os
import hashlib
import yaml


class Crawler:
    """
    Discovers files in configured directories and tracks which ones
    are new or modified using SHA-256 hashing.
    """

    def __init__(self, config_path="config.yaml"):
        """
        Load the config file and store the settings as instance variables.
        """
        config_path = os.path.abspath(config_path)
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        config_dir = os.path.dirname(config_path)
        self.watch_paths = [
            path if os.path.isabs(path) else os.path.normpath(os.path.join(config_dir, path))
            for path in config["watch_paths"]
        ]
        self.include_extensions = config["include_extensions"]
        self.skip_directories = config["skip_directories"]
        data_dir = config["data_dir"]
        self.data_dir = data_dir if os.path.isabs(data_dir) else os.path.normpath(os.path.join(config_dir, data_dir))

    def discover_files(self):
        """
        Walk through all watch_paths recursively and collect every file
        that matches include_extensions, skipping skip_directories.

        Returns:
            list[str] — list of absolute file paths
        """
        results=[]
        for path in self.watch_paths:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    if os.path.splitext(filename)[1] in self.include_extensions:
                        full_path = os.path.join(dirpath, filename)
                        results.append(full_path)
                dirnames[:] = [d for d in dirnames if d not in self.skip_directories]
        return results        

        
    def compute_hash(self, filepath):
        """
        Compute the SHA-256 hash of a file's contents.

        Args:
            filepath (str) — absolute path to the file

        Returns:
            str — hex string of the SHA-256 hash
        """
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()   

    def get_new_and_modified(self, known_hashes=None):
        """
        Compare discovered files against previously known hashes to find
        which files are new or have been modified since last run.

        Args:
            known_hashes (dict) — {filepath: hash} from previous run
                                   Pass None or {} on first run.

        Returns:
            tuple: (files_to_process, current_hashes, deleted_files)
            - files_to_process: list[str] — paths that are new or changed
            - current_hashes: dict — {filepath: hash} for ALL current files
            - deleted files: list[str] — files that were deleted
        """
        if known_hashes is None:
            known_hashes = {}
        current_files = self.discover_files()
        files_to_process = []
        current_hashes = {}
        for file in current_files:
            file_hash = self.compute_hash(file)
            if file not in known_hashes or file_hash != known_hashes[file]:
                files_to_process.append(file)
            current_hashes[file] = file_hash
        
        deleted_files = set(known_hashes.keys()) - set(current_hashes.keys())
        
        return files_to_process, current_hashes, deleted_files


# --- Test it ---
if __name__ == "__main__":
    crawler = Crawler()
    files = crawler.discover_files()
    print(f"Found {len(files)} files:")
    for f in files:
        print(f"  {f}")

    print("\n--- Checking for new/modified ---")
    to_process, hashes = crawler.get_new_and_modified()
    print(f"{len(to_process)} files to process")
