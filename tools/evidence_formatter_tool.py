# evidence_formatter_tool.py - Deprecated
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Note: This tool is deprecated as tools now return EvidenceItem objects directly.
# Consider repurposing or removing this file.

def _deprecated_message():
    print("Warning: 'evidence_formatter_tool.py' is deprecated. Tools now return EvidenceItem objects directly.")

if __name__ == "__main__":
    _deprecated_message()
    