import supervision as sv
from pathlib import Path

if __name__ == "__main__":
    overwrite = False
    image_dir = Path("/media/roel/792678776FE0CD29/hirtshals_2025/golden_standard/")
    ann_dir = Path(
        "/media/roel/792678776FE0CD29/hirtshals_2025/golden_standard/WorkInProgress/20250605"
    )

    for image_subdir in image_dir.glob("20250605-batch*/dtu-zedxmini_right/images"):
        print(f"{image_subdir=}")
        batchname = image_subdir.parent.parent.name
        ann_dir_out = ann_dir / batchname / "darwin"
        if ann_dir_out.exists() and not overwrite:
            print(f"Skipping {batchname}, because we already have {ann_dir_out}")
            continue
        ann_file = ann_dir / batchname / f"{batchname}_annotations.xml"
        d = sv.DetectionDataset.from_cvat_video(
            str(image_subdir),
            str(ann_file),
            attribute_class="fao_code",
        )
        print(d.classes)
        d.as_darwin(
            f"optifish_{batchname}",
            annotations_directory_path=(ann_dir / batchname / "darwin"),
            approximation_percentage=0.9,
        )
    exit()
    # classes = [
    #     "BLL",
    #     "COD",
    #     "DAB",
    #     "FLE",
    #     "GUG",
    #     "HAD",
    #     "HER",
    #     "HKE",
    #     "LEM",
    #     "MON",
    #     "NEP",
    #     "PLA",
    #     "PLE",
    #     "POK",
    #     "SOL",
    #     "SQR",
    #     "WEG",
    #     "WHG",
    # ]
