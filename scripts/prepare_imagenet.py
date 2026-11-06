from __future__ import annotations

import argparse
import os
from pathlib import Path


ARCHIVE_NAMES = (
    "ILSVRC2012_img_train.tar",
    "ILSVRC2012_img_val.tar",
    "ILSVRC2012_devkit_t12.tar.gz",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare an ImageNet-1k ImageFolder layout from the official ILSVRC2012 archives. "
            "This script links the official archives into the output root, optionally links an "
            "already extracted train directory, and uses torchvision's official devkit/val parsing "
            "logic to build val/<wnid>/*.JPEG."
        )
    )
    parser.add_argument(
        "--archive-root",
        required=True,
        help="Directory containing official ILSVRC2012 archives such as ILSVRC2012_img_val.tar.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory that should end up containing train/, val/, meta.bin, and archive symlinks.",
    )
    parser.add_argument(
        "--train-dir",
        default=None,
        help="Optional existing extracted train directory to symlink as output_root/train.",
    )
    parser.add_argument(
        "--skip-val-prepare",
        action="store_true",
        help="Only link archives and train directory; do not parse the validation archive.",
    )
    return parser


def resolve_path(project_root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def count_files(root: Path) -> int:
    return sum(1 for path in root.rglob("*") if path.is_file())


def ensure_symlink(dst: Path, src: Path) -> str:
    if dst.is_symlink():
        if dst.resolve() == src.resolve():
            return "reused_symlink"
        raise FileExistsError(f"Refusing to replace existing symlink: {dst} -> {dst.resolve()}")

    if dst.exists():
        if src.exists() and os.path.samefile(dst, src):
            return "reused_existing"
        return "kept_existing"

    dst.symlink_to(src, target_is_directory=src.is_dir())
    return "created_symlink"


def link_official_archives(archive_root: Path, output_root: Path) -> None:
    for archive_name in ARCHIVE_NAMES:
        src = archive_root / archive_name
        if not src.exists():
            if archive_name == "ILSVRC2012_img_train.tar":
                print(f"archive_missing_optional={src}")
                continue
            raise FileNotFoundError(f"Missing required archive: {src}")
        dst = output_root / archive_name
        status = ensure_symlink(dst, src)
        print(f"archive_link[{archive_name}]={status}")


def maybe_link_train_dir(train_dir: Path | None, output_root: Path) -> None:
    output_train = output_root / "train"
    if train_dir is None:
        if output_train.exists():
            print(f"train_dir_existing={output_train}")
            return
        print("train_dir_missing=output_root/train does not exist and --train-dir was not provided")
        return

    if not train_dir.exists():
        raise FileNotFoundError(f"--train-dir does not exist: {train_dir}")

    status = ensure_symlink(output_train, train_dir)
    print(f"train_link={status}")


def validation_ready(val_dir: Path) -> bool:
    if not val_dir.exists():
        return False
    class_dirs = [path for path in val_dir.iterdir() if path.is_dir()]
    if not class_dirs:
        return False
    return count_files(val_dir) > 0


def prepare_validation_split(output_root: Path) -> None:
    try:
        from torchvision.datasets.imagenet import parse_devkit_archive, parse_val_archive
    except ModuleNotFoundError as exc:
        if exc.name == "scipy":
            raise ModuleNotFoundError(
                "ImageNet devkit parsing requires scipy. Run `pip install -e .` again after pulling "
                "this change, or install scipy into the active environment."
            ) from exc
        raise

    meta_file = output_root / "meta.bin"
    val_dir = output_root / "val"

    if not meta_file.exists():
        print("meta_status=building_meta_from_devkit")
        parse_devkit_archive(str(output_root))
    else:
        print("meta_status=reusing_existing_meta")

    if validation_ready(val_dir):
        print("val_status=already_prepared")
        return

    if val_dir.exists() and any(val_dir.iterdir()):
        raise FileExistsError(
            f"Validation directory exists but does not look fully prepared: {val_dir}. "
            "Please move or remove it before rerunning."
        )

    print("val_status=preparing_from_official_archive")
    parse_val_archive(str(output_root))


def summarize_layout(output_root: Path) -> None:
    train_dir = output_root / "train"
    val_dir = output_root / "val"

    print("imagenet1k_prepare_summary")
    print(f"output_root={output_root}")
    print(f"train_dir={train_dir}")
    print(f"val_dir={val_dir}")
    print(f"meta_file={output_root / 'meta.bin'}")
    print(f"train_exists={train_dir.exists()}")
    print(f"val_exists={val_dir.exists()}")

    if train_dir.exists():
        train_class_dirs = sum(1 for path in train_dir.iterdir() if path.is_dir())
        print(f"train_class_dirs={train_class_dirs}")
    if val_dir.exists():
        val_class_dirs = sum(1 for path in val_dir.iterdir() if path.is_dir())
        print(f"val_class_dirs={val_class_dirs}")
        print(f"val_files_scanned={count_files(val_dir)}")


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    archive_root = resolve_path(project_root, args.archive_root)
    output_root = resolve_path(project_root, args.output_root)
    train_dir = resolve_path(project_root, args.train_dir)
    assert archive_root is not None
    assert output_root is not None

    if not archive_root.exists():
        raise FileNotFoundError(f"Archive root does not exist: {archive_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    link_official_archives(archive_root, output_root)
    maybe_link_train_dir(train_dir, output_root)

    if not args.skip_val_prepare:
        prepare_validation_split(output_root)

    summarize_layout(output_root)


if __name__ == "__main__":
    main()
