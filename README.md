# python-project-template

This is a template repository for Python-based research projects.

## Usage

1. [Create a new repository](https://github.com/allenai/python-project-template/generate) from this template with the desired name of your Python project.

2. Change the name of the `rank_outputs` directory to the name of your repo / Python project.

3. Replace all mentions of `rank_outputs` throughout this repository with the new name.

    On OS X, a quick way to find all mentions of `rank_outputs` is:

    ```bash
    find . -type f -not -path './.git/*' -not -path ./README.md -not -path './docs/build/*' -not -path '*__pycache__*' | xargs grep 'rank_outputs'
    ```

    There is also a one-liner to find and replace all mentions `rank_outputs` with `actual_name_of_project`:

    ```bash
    find . -type f -not -path './.git/*' -not -path ./README.md -not -path './docs/build/*' -not -path '*__pycache__*' -exec sed -i '' -e 's/rank_outputs/actual_name_of_project/' {} \;
    ```

4. Update the README.md.

5. Commit and push your changes, then make sure all CI checks pass.
