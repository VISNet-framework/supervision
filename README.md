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

To run the pre-commit tool, follow these steps:

Install pre-commit by running the following command: `poetry install --with dev`. It will not only install pre-commit but also install all the deps and dev-deps of project

Once pre-commit is installed, navigate to the project's root directory.

Run the command `pre-commit run --all-files`. This will execute the pre-commit hooks configured for this project against the modified files. If any issues are found, the pre-commit tool will provide feedback on how to resolve them. Make the necessary changes and re-run the pre-commit command until all issues are resolved.

You can also install pre-commit as a git hook by executing `pre-commit install`. Every time you do a `git commit` pre-commit run automatically for you.

For more information see: (roboflow/contributing)[https://supervision.roboflow.com/latest/contributing]
