# OpTaS Documentation

The documentation is built as follows.

1. Open a terminal
2. Install doxygen, on Ubuntu: `$ sudo apt install doxygen`
3. Change directory: `$ cd /path/to/optas/doc`
4. Run doxygen `$ doxygen`

If you want to read the documentation in a web browser then open `html/index.html` in Chrome/Firefox/etc.

If you want to read the documentation from a PDF, then you need to compile the documentation as follows.

1. `$ cd /path/to/optas/doc/latex`
2. `$ make`
3. Open the document `refman.pdf`
