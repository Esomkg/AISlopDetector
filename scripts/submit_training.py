"""Submit a training job to Azure ML GPU compute cluster.

Prerequisites:
  1. Azure subscription with AISlopDetector infrastructure deployed
     cd infrastructure/terraform && terraform apply

  2. Azure CLI installed and logged in
     az login

  3. Docker image pushed to ACR
     az acr login --name aislopacr
     docker tag aislop-airflow:latest aislopacr.azurecr.io/aislop:latest
     docker push aislopacr.azurecr.io/aislop:latest

Usage:
  python scripts/submit_training.py                    # defaults
  python scripts/submit_training.py --epochs 20 --spot  # 20 epochs, spot VMs
  python scripts/submit_training.py --dry-run            # validate without submitting
"""

import argparse
import subprocess
import sys


DEFAULTS = {
    "workspace": "aislop-ml",
    "resource_group": "aislop-detector-rg",
    "compute": "gpu-cluster",
    "image": "aislopacr.azurecr.io/aislop:latest",
    "experiment": "aislop-detector",
    "epochs": 10,
    "batch_size": 64,
    "backbone": "efficientnet_b3",
}


def check_prerequisites():
    """Verify Azure CLI and required resources are available."""
    print("Checking prerequisites...")

    try:
        result = subprocess.run(
            ["az", "account", "show", "--query", "name", "-o", "tsv"],
            capture_output=True, text=True, check=True,
        )
        print(f"  Azure login: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  ERROR: Azure CLI not installed or not logged in.")
        print("  Install: https://docs.microsoft.com/cli/azure/install-azure-cli")
        print("  Login:   az login")
        return False

    try:
        subprocess.run(
            ["az", "ml", "workspace", "show",
             "--resource-group", DEFAULTS["resource_group"],
             "--name", DEFAULTS["workspace"]],
            capture_output=True, check=True,
        )
        print(f"  Workspace: {DEFAULTS['workspace']} (OK)")
    except subprocess.CalledProcessError:
        print(f"  ERROR: Workspace '{DEFAULTS['workspace']}' not found.")
        print("  Run: cd infrastructure/terraform && terraform apply")
        return False

    return True


def submit_job(args):
    """Submit the training job via Azure ML CLI."""
    job_file = "infrastructure/kubernetes/azureml/job.yaml"

    if args.dry_run:
        print(f"\n[Dry run] Would submit: az ml job create --file {job_file}")
        return

    print(f"\nSubmitting Azure ML training job...")
    print(f"  Compute:  {args.compute}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  Spot VMs: {args.spot}")
    print(f"  Image:    {args.image}")

    cmd = [
        "az", "ml", "job", "create",
        "--file", job_file,
        "--resource-group", args.resource_group,
        "--workspace-name", args.workspace,
        "--set", f"compute=azureml:{args.compute}",
        "--set", f"command=python src/models/azureml_train.py --data-dir data/raw/cifake --epochs {args.epochs} --batch-size {args.batch_size} --backbone {args.backbone} --download",
        "--set", f"environment.image={args.image}",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"\n{result.stdout}")

        import json
        job_info = json.loads(result.stdout)
        job_name = job_info.get("name", "unknown")
        print(f"\nJob submitted: {job_name}")
        print(f"Monitor: az ml job show --name {job_name} -g {args.resource_group} -w {args.workspace}")
        print(f"Stream:  az ml job stream --name {job_name} -g {args.resource_group} -w {args.workspace}")
        print(f"\nNote: GPU node auto-scales from 0 — first run takes ~5 min to provision.")

    except subprocess.CalledProcessError as e:
        print(f"\nERROR submitting job:\n{e.stderr}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Submit AISlopDetector training to Azure ML GPU cluster"
    )
    parser.add_argument("--workspace", default=DEFAULTS["workspace"])
    parser.add_argument("--resource-group", default=DEFAULTS["resource_group"])
    parser.add_argument("--compute", default=DEFAULTS["compute"])
    parser.add_argument("--image", default=DEFAULTS["image"])
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--backbone", default=DEFAULTS["backbone"])
    parser.add_argument("--spot", action="store_true", help="Use spot VMs (cheaper)")
    parser.add_argument("--dry-run", action="store_true", help="Validate without submitting")
    args = parser.parse_args()

    if not args.dry_run and not check_prerequisites():
        sys.exit(1)

    submit_job(args)


if __name__ == "__main__":
    main()
