"""Upload the folders built by scripts/build_hf_release.py. Run this yourself.

    huggingface-cli login
    python scripts/upload_to_hub.py --model                      # model repo only
    python scripts/upload_to_hub.py --space atilimai/plantvillage-demo

Only do this after RELEASE_CHECKLIST.md is signed off.
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

RELEASE = Path(__file__).resolve().parents[1] / "release" / "huggingface"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="store_true", help="upload release/huggingface/model")
    parser.add_argument("--model-repo", default="atilimai/plantvillage-leaf-disease-classifier")
    parser.add_argument("--space", metavar="REPO_ID", help="also create/update a Gradio Space")
    parser.add_argument("--message", default="Release v1.0.0: leaf-level leakage-free models")
    args = parser.parse_args()

    api = HfApi()
    if args.model:
        api.create_repo(args.model_repo, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=RELEASE / "model", repo_id=args.model_repo, repo_type="model",
                          commit_message=args.message)
        print(f"https://huggingface.co/{args.model_repo}")
    if args.space:
        api.create_repo(args.space, repo_type="space", space_sdk="gradio", exist_ok=True)
        api.upload_folder(folder_path=RELEASE / "space", repo_id=args.space, repo_type="space",
                          commit_message=args.message)
        print(f"https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
