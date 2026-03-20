# searcher/facet_filter.py

import os
from datetime import datetime


class FacetFilter:
    """
    Post-retrieval filtering by structured metadata.
    Applied AFTER fusion and reranking to restrict results
    without hurting semantic quality.

    Supported facets:
        file_type   — e.g. [".pdf", ".docx"]
        date_after  — only files modified after this date
        date_before — only files modified before this date
        min_size    — minimum file size in bytes
        max_size    — maximum file size in bytes
        directory   — only files inside this directory path
    """

    def filter(
        self,
        results: list[dict],
        file_type: list[str] = None,
        date_after: datetime = None,
        date_before: datetime = None,
        min_size: int = None,
        max_size: int = None,
        directory: str = None,
    ) -> list[dict]:
        """
        Apply facet filters to a list of search results.

        Args:
            results     — list of chunk dicts (must include 'filepath')
            file_type   — list of extensions e.g. [".pdf", ".txt"]
            date_after  — datetime object; exclude files older than this
            date_before — datetime object; exclude files newer than this
            min_size    — minimum file size in bytes
            max_size    — maximum file size in bytes
            directory   — restrict to files under this directory

        Returns:
            list[dict] — filtered results (same structure, subset)
        """
        filtered = []

        for result in results:
            fp = result["filepath"]

            # File type filter
            if file_type:
                ext = os.path.splitext(fp)[1].lower()
                if ext not in [e.lower() for e in file_type]:
                    continue

            # Directory scope filter
            if directory:
                if not fp.startswith(os.path.abspath(directory)):
                    continue

            # File system checks (only if file still exists)
            if os.path.exists(fp):
                stat = os.stat(fp)

                # Date filters
                mtime = datetime.fromtimestamp(stat.st_mtime)
                if date_after and mtime < date_after:
                    continue
                if date_before and mtime > date_before:
                    continue

                # Size filters
                size = stat.st_size
                if min_size and size < min_size:
                    continue
                if max_size and size > max_size:
                    continue

            filtered.append(result)

        return filtered