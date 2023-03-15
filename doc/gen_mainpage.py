import re
import pathlib


def fix_line(line):
    # Define a regular expression pattern to match the inline LaTeX
    pattern = r"\$(.*?)\$"

    # Define a function to handle the replacement
    def replace_latex(match):
        return r"\f$%s\f$" % match.group(1)

    if "$" in line:  # and ('$$' not in line):
        line = re.sub(pattern, replace_latex, line)

    return line


def main():
    repo_path = pathlib.Path(__file__).parent.absolute().parent.absolute()
    with open(repo_path / "README.md", "r") as input_file:
        with open("mainpage.md", "w") as output_file:
            for line in input_file.readlines():
                new_line = fix_line(line)
                output_file.write(new_line)


if __name__ == "__main__":
    main()
