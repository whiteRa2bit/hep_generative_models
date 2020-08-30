import numpy as np
import uproot

from generation.config import ROOT_FILE_PATH, ROOT_TREE_NAME, ATTRIBUTES, ATTRIBUTE_PATHS


def _process_attribute(root_tree, attr, attr_path):
    print(f"Processing attribute {attr}...")
    attr_values = root_tree.array(attr)
    np.save(attr_path, attr_values)


def main():
    root_file = uproot.open(ROOT_FILE_PATH)
    root_tree = root_file[ROOT_TREE_NAME]

    for attr, attr_path in zip(ATTRIBUTES, ATTRIBUTE_PATHS):
        _process_attribute(root_tree, attr, attr_path)


if __name__ == '__main__':
    main()
