# /// script
# dependencies = [
# ]
# ///

"""
Script to build and push Docker images.

This differs from the script in gradium-serve in that it uses the
dockerfile parent directory as context rather than the root directory.
"""

import argparse
from dataclasses import dataclass
import pathlib
import subprocess


@dataclass
class Image:
    name: str
    dockerfile: pathlib.Path
    context: pathlib.Path


NAME_MAPPINGS = {
    "": "unmute-backend",
    "frontend": "unmute-frontend",
}


def map_name(name: str) -> str:
    if name in NAME_MAPPINGS:
        return NAME_MAPPINGS[name]
    raise ValueError(f"Unknown image name mapping for '{name}'")


def run_command(cmd: list[str], cwd: pathlib.Path | None = None) -> int:
    """Run a command and return exit code."""
    print(f"Running: {' '.join(cmd)}")
    if cwd is not None:
        print(f"  in directory: {cwd}")

    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
        return result.returncode

    print("✅ Command succeeded")
    return 0


def check_docker_login(registry: str) -> bool:
    """Check if user is logged into the registry."""
    print(f"🔐 Checking Docker login for {registry}...")

    # Try to get login info for the registry
    result = subprocess.run(
        ["docker", "system", "info", "--format", "json"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("❌ Could not check Docker status")
        return False

    # For Scaleway registry, check if we can authenticate
    registry_host = registry.split("/")[0]

    # Try a simple approach - attempt to pull a dummy image
    auth_check = subprocess.run(
        ["docker", "pull", f"{registry}/nonexistent:test"],
        capture_output=True,
        text=True,
    )

    # If we get authentication error, user is not logged in
    keywords = ("authentication", "unauthorized", "denied")
    if any(keyword in auth_check.stderr.lower() for keyword in keywords):
        print(f"❌ Not logged into {registry}")
        print("Please run:")
        print(f"  docker login {registry_host}")
        return False

    print(f"✅ Docker login verified for {registry}")
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build and push API and Worker Docker images"
    )
    parser.add_argument(
        "--tag", default=None, help="Tag for the images (e.g., latest, v1.0.0)"
    )
    parser.add_argument(
        "--registry",
        default="rg.fr-par.scw.cloud/gradium",
        help="Docker registry (default: rg.fr-par.scw.cloud/gradium)",
    )
    parser.add_argument(
        "folders",
        nargs="*",
        help="Folders containing Dockerfiles to build (default: api "
        "compute-engine/worker)",
    )

    args = parser.parse_args()

    # Check if tag is provided
    if args.tag is None:
        print("❌ Error: --tag is required")
        print("Please provide a tag for the images (e.g., --tag latest)")
        return

    # Project root directory
    root_dir = None
    for parent in pathlib.Path(__file__).parents:
        if (parent / ".git").exists():
            root_dir = parent
            break
    if root_dir is None:
        print("❌ Error: no root directory found")
        return

    # Default folders if none specified
    if not args.folders:
        # Find all Dockerfile locations recursively
        dockerfiles = list(root_dir.glob("**/Dockerfile"))
        args.folders = [str(df.parent.relative_to(root_dir)) for df in dockerfiles]

    # Image configurations
    images = []
    for folder in args.folders:
        dockerfile = root_dir / folder / "Dockerfile"
        if not dockerfile.exists():
            print(f"⚠️ Dockerfile not found: {dockerfile}, skipping...")
            continue
        # Image name is the last part of the folder path
        images.append(
            Image(
                name=map_name(pathlib.Path(folder).name),
                dockerfile=dockerfile,
                context=root_dir / folder,
            )
        )

    # Check if we have any images to build
    if not images:
        print("❌ No valid Dockerfiles found")
        return

    print(f"🚀 Building and pushing images to {args.registry}")
    print(f"🏷️ Using tag: {args.tag}")
    print(f"📦 Found {len(images)} image(s) to build")

    # Build all images first
    print("\n🔨 Building all images...")
    for image_config in images:
        full_image_name = f"{args.registry}/{image_config.name}:{args.tag}"

        print(f"\n🔨 Building {image_config.name}...")
        print(f"📁 Context: {image_config.context}")
        print(f"📄 Dockerfile: {image_config.dockerfile}")
        print(f"🏷️ Tag: {full_image_name}")

        build_cmd = [
            "docker",
            "build",
            "--network=host",
            "-f",
            str(image_config.dockerfile),
            "-t",
            full_image_name,
            str(image_config.context),
        ]

        if run_command(build_cmd) != 0:
            print(f"❌ Failed to build {image_config.name}")
            return

    # Check Docker login before pushing
    if not check_docker_login(args.registry):
        return

    # Push all images
    print("\n📤 Pushing all images...")
    for image_config in images:
        full_image_name = f"{args.registry}/{image_config.name}:{args.tag}"

        print(f"\n📤 Pushing {full_image_name}...")
        push_cmd = ["docker", "push", full_image_name]

        if run_command(push_cmd) != 0:
            print(f"❌ Failed to push {image_config.name}")
            return

    success = True

    if success:
        print("\n🎉 All images built and pushed successfully!")
        print("\nImage names:")
        for image_config in images:
            print(f"  {args.registry}/{image_config.name}:{args.tag}")
    else:
        print("\n❌ Failed to build/push images")
        return


if __name__ == "__main__":
    main()
