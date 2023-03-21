import re
import pathlib

"""

This is a very naive parser that converts the README.md file, accepted
by GitHub, to the doc/mainpage.md file, accepted by doxygen.  It is
possible that it may get some things wrong, so always a good idea to
build locally first and check in a browser.

I haven't found an alternative robust solution yet so this "hacky"
script will have to do for now.

"""


class LineFixer:
    def __init__(self):
        self._outside_formula = True
        self._outside_code = True

    def fix(self, line):
        # Adjust width of youtube video image
        if 'width="50%"' in line and "REBmbCANx0s" in line:
            return line.replace('width="50%"', 'width="30%"')

        # Check if this is the start/end of code
        if ("```python" in line) or "```" in line or ("```bibtex" in line):
            if self._outside_code:
                # we are outside code extract (i.e. at the start), so replace with the start of code
                if "python" in line:
                    line = "\code{.py}\n"  # need \n since a space is put at the start of the code snippet otherwise
                elif "bibtex" in line:
                    line = "\code{.bib}\n"  # need \n since a space is put at the start of the code snippet otherwise
            else:
                # we are inside code extract (i.e. at the end), so replace with the end of code
                line = "\endcode"
            self._outside_code = not self._outside_code
            return line

        # Check if ` character appears twice in the line (i.e. inline code)
        if line.count("`") == 2:
            # Define a regular expression pattern to match the inline LaTeX
            pattern = r"\`(.*?)\`"

            # Define a function to handle the replacement
            def replace_latex(match):
                return r"<code>%s</code>" % match.group(1)

            line = re.sub(pattern, replace_latex, line)

            return line

        # Check if this line corresponds to beginning/ending of a formula
        if "$$" in line:
            if self._outside_formula:
                # we are outside the formula (i.e. at the start), so replace with the start of formula
                line = line.replace("$$", r"\f[")
            else:
                # we are inside the formula, so replace with end of formula
                line = line.replace("$$", r"\f]")
            self._outside_formula = not self._outside_formula
            return line

        # Define a regular expression pattern to match the inline LaTeX
        pattern = r"\$(.*?)\$"

        # Define a function to handle the replacement
        def replace_latex(match):
            return r"\f$%s\f$" % match.group(1)

        if "$" in line:  # and ('$$' not in line):
            line = re.sub(pattern, replace_latex, line)

        line = line.replace("\|\|", "\|")

        return line


def main():
    line_fixer = LineFixer()

    repo_path = pathlib.Path(__file__).parent.absolute().parent.absolute()
    doc_path = repo_path / "doc"
    readme_file_name = repo_path / "README.md"
    mainpage_file_name = doc_path / "mainpage.md"

    if mainpage_file_name.is_file():
        mainpage_file_name.unlink()
        print("Removed old version of doc/mainpage.md")

    with open(readme_file_name, "r") as input_file:
        with open(mainpage_file_name, "w") as output_file:
            for line in input_file.readlines():
                new_line = line_fixer.fix(line)
                output_file.write(new_line)

    print("Created doc/mainpage.md")


if __name__ == "__main__":
    main()
