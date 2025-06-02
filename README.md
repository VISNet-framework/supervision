# Supervision fork
Repository fork of (supervision)[https://github.com/roboflow/supervision].
Added features are:
  - Darwin support

## Edited files
- Workflows
- README.md
- .pre-commit-config.yaml

## For contributors
This project uses the (pre-commit tool)[https://pre-commit.com/] to maintain code quality and consistency. Before submitting a pull request or making any commits, it is important to run the pre-commit tool to ensure that your changes meet the project's guidelines.

Furthermore, we have integrated a pre-commit GitHub Action into our workflow. This means that with every pull request opened, the pre-commit checks will be automatically enforced, streamlining the code review process and ensuring that all contributions adhere to our quality standards.

To install the development packages, including pre-commit, follow these steps:

```
python -m venv venv
source venv/bin/activate
pip install -e .[dev]
pre-commit install
```

Every time you do a `git commit` pre-commit runss automatically for you.

For more information see: (roboflow/contributing)[https://supervision.roboflow.com/latest/contributing]
