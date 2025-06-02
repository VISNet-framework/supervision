from supervision.dataset.core import DetectionDataset

if __name__ == "__main__":
    from pathlib import Path

    base_dir = "/mnt/wur-w-VisionRoboticsData/GARdata/new_format/0000000000_sample/datasets/"
    images_directory_path = base_dir + "images/"
    annotations_path = base_dir + "anns/0-insseg-sample/darwin"

    force_masks: bool = (False,)
    classes = [
        "rose",
    ]
    dataset = DetectionDataset.from_darwin(
        images_directory_path=images_directory_path,
        annotations_path=annotations_path,
        classes=classes,
    )

    images_directory_path_output = Path(
        "testing_superivision_woohoo/images/"
    )
    annotations_path_output = Path(
        "testing_superivision_woohoo/annotations/0-insseg-sample/darwin"
    )
    dataset.as_darwin(
        darwin_dataset_name="testing_supervision_woohoo",
        images_directory_path=images_directory_path_output,
        annotations_directory_path=annotations_path_output,
    )
    print("done")
