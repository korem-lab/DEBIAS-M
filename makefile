all: _site.yml
	Rscript -e "rmarkdown::render('index.Rmd')"
	Rscript -e "rmarkdown::render('DebiasMClassifier.Rmd')"
	Rscript -e "rmarkdown::render('DebiasMRegressor.Rmd')"
	Rscript -e "rmarkdown::render('MultitaskDebiasMClassifier.Rmd')"
	Rscript -e "rmarkdown::render('MultitaskDebiasMRegressor.Rmd')"
	Rscript -e "rmarkdown::render('OnlineDebiasMClassifier.Rmd')"
	Rscript -e "rmarkdown::render('DebiasMClassifierLogAdd.Rmd')"
