MAIN = main
LATEX = pdflatex
FLAGS = -interaction=nonstopmode -halt-on-error

.PHONY: all clean distclean

all:
	$(LATEX) $(FLAGS) $(MAIN).tex
	$(LATEX) $(FLAGS) $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).toc $(MAIN).out $(MAIN).lof $(MAIN).lot

distclean: clean
	rm -f $(MAIN).pdf
